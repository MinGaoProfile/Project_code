from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from torch.utils.data import Sampler


class BalancedPatchBatchSampler(Sampler[list[int]]):
    """Build exact 1:1 positive/negative batches from stratified patch pools."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        config: dict[str, Any],
        *,
        seed: int,
        training: bool,
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced patch sampling requires an even batch_size >= 2")
        if not dataset.positive_indices:
            raise ValueError("Balanced patch sampling requires positive patches")
        if not any(dataset.negative_indices_by_category.values()):
            raise ValueError("Balanced patch sampling requires pure-negative patches")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 2
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        self.negative_categories = tuple(dataset.negative_indices_by_category)
        positive_fraction = float(config.get("positive_fraction", 0.5))
        if not math.isclose(positive_fraction, 0.5):
            raise ValueError("Exact balanced sampling requires positive_fraction: 0.5")
        self.category_fractions = {
            str(name): float(value)
            for name, value in config.get(
                "negative_category_fractions",
                {"adjacent": 0.3, "anatomical_coverage": 0.7},
            ).items()
        }
        if set(self.category_fractions) != set(self.negative_categories):
            raise ValueError(
                "negative_category_fractions must match the dataset negative categories: "
                f"{', '.join(self.negative_categories)}"
            )
        total = sum(self.category_fractions.values())
        if total <= 0:
            raise ValueError("negative_category_fractions must sum to a positive value")
        self.category_fractions = {
            name: value / total for name, value in self.category_fractions.items()
        }
        self.positive_category_fractions = {
            str(name): float(value)
            for name, value in config.get(
                "positive_category_fractions",
                {"complete": 0.55, "boundary": 0.45},
            ).items()
        }
        if set(self.positive_category_fractions) != {"complete", "boundary"}:
            raise ValueError(
                "positive_category_fractions must define complete and boundary"
            )
        positive_total = sum(self.positive_category_fractions.values())
        if positive_total <= 0:
            raise ValueError(
                "positive_category_fractions must sum to a positive value"
            )
        self.positive_category_fractions = {
            name: value / positive_total
            for name, value in self.positive_category_fractions.items()
        }
        self.positive_scan_group_fractions = {
            str(name): float(value)
            for name, value in config.get(
                "positive_scan_group_fractions",
                {"small_only": 0.7, "large_containing": 0.3},
            ).items()
        }
        if set(self.positive_scan_group_fractions) != {
            "small_only",
            "large_containing",
        }:
            raise ValueError(
                "positive_scan_group_fractions must define small_only and "
                "large_containing"
            )
        group_total = sum(self.positive_scan_group_fractions.values())
        if group_total <= 0:
            raise ValueError(
                "positive_scan_group_fractions must sum to a positive value"
            )
        self.positive_scan_group_fractions = {
            name: value / group_total
            for name, value in self.positive_scan_group_fractions.items()
        }
        self.hard_negative_fraction = float(config.get("hard_negative_fraction", 0.5))
        if not 0.0 <= self.hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0, 1]")
        self.hard_positive_fraction = float(config.get("hard_positive_fraction", 0.3))
        if not 0.0 <= self.hard_positive_fraction <= 1.0:
            raise ValueError("hard_positive_fraction must be in [0, 1]")
        self.maximum_positive_views_per_lesion_per_epoch = int(
            config.get("maximum_positive_views_per_lesion_per_epoch", 2)
        )
        if self.maximum_positive_views_per_lesion_per_epoch < 1:
            raise ValueError(
                "maximum_positive_views_per_lesion_per_epoch must be positive"
            )
        self.hard_negative_scores: dict[int, float] = {}
        self.hard_positive_scores: dict[int, float] = {}
        hard_positive = config.get("hard_positive_mining", {})
        self.maximum_hard_positive_views_per_lesion_per_epoch = int(
            hard_positive.get(
                "maximum_views_per_lesion_per_epoch",
                self.maximum_positive_views_per_lesion_per_epoch + 1,
            )
        )
        if (
            self.maximum_hard_positive_views_per_lesion_per_epoch
            < self.maximum_positive_views_per_lesion_per_epoch
        ):
            raise ValueError(
                "hard-positive maximum_views_per_lesion_per_epoch cannot be "
                "smaller than the ordinary positive-view limit"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_hard_negative_scores(self, scores: dict[int, float]) -> None:
        valid = set(self.all_negative_indices)
        self.hard_negative_scores = {
            int(index): float(score)
            for index, score in scores.items()
            if int(index) in valid and math.isfinite(float(score))
        }

    def set_hard_positive_scores(self, scores: dict[int, float]) -> None:
        valid = set(self.dataset.positive_indices)
        self.hard_positive_scores = {
            int(index): float(score)
            for index, score in scores.items()
            if int(index) in valid and math.isfinite(float(score))
        }

    def update_hard_positive_scores(
        self,
        scores: dict[int, float],
        evaluated_indices: Sequence[int],
        maximum_pool_size: int,
        *,
        score_decay: float = 1.0,
        minimum_retained_score: float = 0.0,
        maximum_per_case: int = 0,
        maximum_per_lesion: int = 0,
    ) -> None:
        """Update low-confidence positive views while preserving case/lesion diversity."""
        valid = set(self.dataset.positive_indices)
        score_decay = float(score_decay)
        if not 0.0 <= score_decay <= 1.0:
            raise ValueError("hard-positive score_decay must be in [0, 1]")
        minimum_retained_score = float(minimum_retained_score)
        retained = {
            index: score * score_decay
            for index, score in self.hard_positive_scores.items()
            if index in valid and score * score_decay >= minimum_retained_score
        }
        for index in evaluated_indices:
            retained.pop(int(index), None)
        retained.update(
            {
                int(index): float(score)
                for index, score in scores.items()
                if int(index) in valid and math.isfinite(float(score))
            }
        )
        ranked = sorted(retained.items(), key=lambda item: item[1], reverse=True)
        lesion_metadata = getattr(self.dataset, "patch_request_lesions", None)
        case_counts: defaultdict[int, int] = defaultdict(int)
        lesion_counts: defaultdict[tuple[int, str, int], int] = defaultdict(int)
        diverse: list[tuple[int, float]] = []
        for index, score in ranked:
            case_index = int(self.dataset.patch_requests[index][0])
            if maximum_per_case > 0 and case_counts[case_index] >= maximum_per_case:
                continue
            lesion = None if lesion_metadata is None else lesion_metadata[index]
            if lesion is not None:
                lesion_key = (int(lesion[0]), str(lesion[1]), int(lesion[2]))
                if maximum_per_lesion > 0 and lesion_counts[lesion_key] >= maximum_per_lesion:
                    continue
                lesion_counts[lesion_key] += 1
            diverse.append((index, score))
            case_counts[case_index] += 1
        if maximum_pool_size > 0:
            diverse = diverse[: int(maximum_pool_size)]
        self.hard_positive_scores = dict(diverse)

    def update_hard_negative_scores(
        self,
        scores: dict[int, float],
        evaluated_indices: Sequence[int],
        maximum_pool_size: int,
        *,
        score_decay: float = 1.0,
        minimum_retained_score: float = 0.0,
        maximum_per_case: int = 0,
    ) -> None:
        """Update a diverse pool while aging stale and rescored-clean entries."""
        valid = set(self.all_negative_indices)
        score_decay = float(score_decay)
        if not 0.0 <= score_decay <= 1.0:
            raise ValueError("hard-negative score_decay must be in [0, 1]")
        minimum_retained_score = float(minimum_retained_score)
        self.hard_negative_scores = {
            index: score * score_decay
            for index, score in self.hard_negative_scores.items()
            if index in valid and score * score_decay >= minimum_retained_score
        }
        for index in evaluated_indices:
            index = int(index)
            if index in valid:
                self.hard_negative_scores.pop(index, None)
        self.hard_negative_scores.update(
            {
                int(index): float(score)
                for index, score in scores.items()
                if int(index) in valid and math.isfinite(float(score))
            }
        )

        # Spatial phases can produce several coordinates for one anatomical slot.
        # Retaining only its strongest false positive prevents duplicate patches from
        # consuming the finite pool.
        best_by_slot: dict[object, tuple[int, float]] = {}
        for index, score in self.hard_negative_scores.items():
            slot = self.dataset.patch_request_slots[index]
            key: object = ("index", index) if slot is None else ("slot", *slot)
            previous = best_by_slot.get(key)
            if previous is None or score > previous[1]:
                best_by_slot[key] = (index, score)
        ranked = sorted(
            best_by_slot.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        maximum_per_case = int(maximum_per_case)
        if maximum_per_case > 0:
            case_counts: defaultdict[int, int] = defaultdict(int)
            diverse: list[tuple[int, float]] = []
            for index, score in ranked:
                case_index = int(self.dataset.patch_requests[index][0])
                if case_counts[case_index] >= maximum_per_case:
                    continue
                diverse.append((index, score))
                case_counts[case_index] += 1
            ranked = diverse

        maximum_pool_size = int(maximum_pool_size)
        if maximum_pool_size > 0:
            ranked = ranked[:maximum_pool_size]
        self.hard_negative_scores = dict(ranked)

    @property
    def all_negative_indices(self) -> list[int]:
        return [
            index
            for category in self.negative_categories
            for index in self.dataset.negative_indices_by_category[category]
        ]

    def _rank_hard_positives_by_case(self) -> list[int]:
        by_case: defaultdict[int, list[int]] = defaultdict(list)
        for index in self.hard_positive_scores:
            record_index = int(self.dataset.patch_requests[index][0])
            by_case[record_index].append(index)
        for indices in by_case.values():
            indices.sort(
                key=lambda index: self.hard_positive_scores[index], reverse=True
            )
        cases = sorted(
            by_case,
            key=lambda case: self.hard_positive_scores[by_case[case][0]],
            reverse=True,
        )
        ordered: list[int] = []
        while any(by_case.values()):
            for case in cases:
                if by_case[case]:
                    ordered.append(by_case[case].pop(0))
        return ordered

    def hard_positive_candidates(
        self,
        limit: int,
        refresh_index: int,
        recheck_fraction: float = 0.5,
    ) -> list[int]:
        """Mix hard-positive rechecks with lesion/case-balanced discovery."""
        all_candidates = list(self.dataset.positive_indices)
        if not all_candidates:
            return []
        limit = len(all_candidates) if limit <= 0 else min(int(limit), len(all_candidates))
        recheck_fraction = float(recheck_fraction)
        if not 0.0 <= recheck_fraction <= 1.0:
            raise ValueError("hard-positive recheck_fraction must be in [0, 1]")
        rng = np.random.default_rng(self.seed + 8191 + int(refresh_index))
        ranked = self._rank_hard_positives_by_case()
        recheck_count = min(len(ranked), int(round(limit * recheck_fraction)))
        selected: list[int] = []
        if recheck_count:
            start = (int(refresh_index) * recheck_count) % len(ranked)
            rotated = ranked[start:] + ranked[:start]
            selected.extend(rotated[:recheck_count])
        used = set(selected)
        current_pool = set(self.hard_positive_scores)
        discovery = [index for index in all_candidates if index not in current_pool]
        discovery = self._round_robin_positive_lesions(discovery, rng)
        for index in discovery:
            if index not in used:
                selected.append(index)
                used.add(index)
            if len(selected) == limit:
                break
        if len(selected) < limit:
            fallback = self._round_robin_positive_lesions(all_candidates, rng)
            for index in fallback:
                if index not in used:
                    selected.append(index)
                    used.add(index)
                if len(selected) == limit:
                    break
        return selected

    def _category_counts(self, count: int) -> dict[str, int]:
        """Allocate an exact count while preserving configured category fractions."""
        raw = {
            category: count * self.category_fractions[category]
            for category in self.negative_categories
        }
        allocated = {category: int(math.floor(value)) for category, value in raw.items()}
        remainder = count - sum(allocated.values())
        ranked = sorted(
            self.negative_categories,
            key=lambda category: raw[category] - allocated[category],
            reverse=True,
        )
        for category in ranked[:remainder]:
            allocated[category] += 1
        return allocated

    def _rank_hard_negatives_by_case(self) -> list[int]:
        by_case: defaultdict[int, list[int]] = defaultdict(list)
        for index in self.hard_negative_scores:
            record_index = int(self.dataset.patch_requests[index][0])
            by_case[record_index].append(index)
        for indices in by_case.values():
            indices.sort(
                key=lambda index: self.hard_negative_scores[index],
                reverse=True,
            )
        cases = sorted(
            by_case,
            key=lambda case: self.hard_negative_scores[by_case[case][0]],
            reverse=True,
        )
        ordered: list[int] = []
        while any(by_case.values()):
            for case in cases:
                if by_case[case]:
                    ordered.append(by_case[case].pop(0))
        return ordered

    def hard_negative_candidates(
        self,
        limit: int,
        refresh_index: int,
        recheck_fraction: float = 0.25,
    ) -> list[int]:
        """Mix stale-pool rechecks with case/category-balanced new discovery."""
        all_candidates = self.all_negative_indices
        if not all_candidates:
            return []
        limit = len(all_candidates) if limit <= 0 else min(int(limit), len(all_candidates))
        recheck_fraction = float(recheck_fraction)
        if not 0.0 <= recheck_fraction <= 1.0:
            raise ValueError("hard-negative recheck_fraction must be in [0, 1]")
        rng = np.random.default_rng(self.seed + int(refresh_index))
        used_indices: set[int] = set()
        used_slots: set[tuple[int, int, int]] = set()
        selected: list[int] = []

        hard_ranked = self._rank_hard_negatives_by_case()
        recheck_count = min(
            len(hard_ranked),
            int(round(limit * recheck_fraction)),
        )
        if recheck_count:
            start = (int(refresh_index) * recheck_count) % len(hard_ranked)
            rotated = hard_ranked[start:] + hard_ranked[:start]
            selected.extend(
                self._take_unique_slots(
                    rotated,
                    recheck_count,
                    used_indices,
                    used_slots,
                )
            )

        discovery_target = limit - len(selected)
        current_pool = set(self.hard_negative_scores)
        category_counts = self._category_counts(discovery_target)
        for category in self.negative_categories:
            candidates = [
                index
                for index in self.dataset.negative_indices_by_category[category]
                if index not in current_pool
            ]
            candidates = self._round_robin_cases(candidates, rng)
            selected.extend(
                self._take_unique_slots(
                    candidates,
                    category_counts[category],
                    used_indices,
                    used_slots,
                )
            )

        if len(selected) < limit:
            fallback = self._round_robin_cases(all_candidates, rng)
            selected.extend(
                self._take_unique_slots(
                    fallback,
                    limit - len(selected),
                    used_indices,
                    used_slots,
                )
            )
        return selected

    def __len__(self) -> int:
        capacity = self._positive_epoch_capacity()
        if capacity < self.per_class:
            raise ValueError(
                "Positive lesion-view cap leaves fewer samples than one balanced batch"
            )
        return capacity // self.per_class

    def _positive_epoch_capacity(self) -> int:
        lesion_metadata = getattr(self.dataset, "patch_request_lesions", None)
        if lesion_metadata is None:
            return len(self.dataset.positive_indices)
        counts: defaultdict[tuple[int, str, int], int] = defaultdict(int)
        ungrouped = 0
        for index in self.dataset.positive_indices:
            group = lesion_metadata[index]
            if group is None:
                ungrouped += 1
            else:
                record_index, guidance_name, component_id = group
                counts[(int(record_index), str(guidance_name), int(component_id))] += 1
        maximum = self.maximum_positive_views_per_lesion_per_epoch
        return ungrouped + sum(min(count, maximum) for count in counts.values())

    def _round_robin_cases(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> list[int]:
        by_case: defaultdict[int, list[int]] = defaultdict(list)
        for index in indices:
            record_index = int(self.dataset.patch_requests[index][0])
            by_case[record_index].append(int(index))
        cases = list(by_case)
        if self.training:
            rng.shuffle(cases)
            for values in by_case.values():
                rng.shuffle(values)
        ordered: list[int] = []
        while any(by_case.values()):
            for case in cases:
                if by_case[case]:
                    ordered.append(by_case[case].pop())
        return ordered

    def _round_robin_positive_lesions(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> list[int]:
        lesion_metadata = getattr(self.dataset, "patch_request_lesions", None)
        if lesion_metadata is None:
            return self._round_robin_cases(indices, rng)
        by_case: defaultdict[int, dict[tuple[str, int], list[int]]] = defaultdict(dict)
        for index in indices:
            group = lesion_metadata[index]
            if group is None:
                continue
            record_index, guidance_name, component_id = group
            lesion = (str(guidance_name), int(component_id))
            by_case[int(record_index)].setdefault(lesion, []).append(int(index))
        if not by_case:
            return self._round_robin_cases(indices, rng)

        cases = list(by_case)
        if self.training:
            rng.shuffle(cases)
        queues: dict[int, list[list[int]]] = {}
        for case, lesions in by_case.items():
            lesion_names = list(lesions)
            if self.training:
                rng.shuffle(lesion_names)
                for values in lesions.values():
                    rng.shuffle(values)
            queues[case] = [lesions[name] for name in lesion_names]

        ordered: list[int] = []
        while any(queues.values()):
            for case in cases:
                case_queues = queues[case]
                while case_queues and not case_queues[0]:
                    case_queues.pop(0)
                if not case_queues:
                    continue
                lesion_queue = case_queues.pop(0)
                ordered.append(lesion_queue.pop())
                if lesion_queue:
                    case_queues.append(lesion_queue)
        return ordered

    def _repeat_positive_candidates(
        self,
        candidates: Sequence[int],
        count: int,
        rng: np.random.Generator,
        lesion_counts: defaultdict[tuple[int, str, int], int] | None = None,
        maximum_per_lesion: int | None = None,
        preserve_order: bool = False,
    ) -> list[int]:
        if count <= 0 or not candidates:
            return []
        ordered = (
            [int(index) for index in candidates]
            if preserve_order
            else self._round_robin_positive_lesions(candidates, rng)
        )
        if lesion_counts is None:
            return [ordered[index % len(ordered)] for index in range(count)]
        lesion_metadata = self.dataset.patch_request_lesions
        maximum = (
            self.maximum_positive_views_per_lesion_per_epoch
            if maximum_per_lesion is None
            else int(maximum_per_lesion)
        )
        selected: list[int] = []
        while len(selected) < count:
            added = False
            for index in ordered:
                group = lesion_metadata[index]
                if group is None:
                    selected.append(index)
                    added = True
                else:
                    record_index, guidance_name, component_id = group
                    key = (int(record_index), str(guidance_name), int(component_id))
                    if (
                        lesion_counts[key]
                        >= maximum
                    ):
                        continue
                    selected.append(index)
                    lesion_counts[key] += 1
                    added = True
                if len(selected) == count:
                    break
            if not added:
                break
        return selected

    def _select_positive_subset(
        self,
        candidates: Sequence[int],
        count: int,
        rng: np.random.Generator,
        lesion_counts: defaultdict[tuple[int, str, int], int] | None,
    ) -> list[int]:
        categories = getattr(self.dataset, "positive_indices_by_category", None)
        candidates = list(candidates)
        if count <= 0 or not candidates:
            return []
        if not categories or not any(categories.values()):
            return self._repeat_positive_candidates(
                candidates, count, rng, lesion_counts
            )
        allowed = set(candidates)
        complete = [
            index for index in categories.get("complete", []) if index in allowed
        ]
        boundary = [
            index for index in categories.get("boundary", []) if index in allowed
        ]
        if not complete:
            complete_count = 0
        elif not boundary:
            complete_count = count
        else:
            complete_count = int(
                round(count * self.positive_category_fractions["complete"])
            )
        boundary_count = count - complete_count
        # Reserve one boundary/context view per lesion before complete views can use
        # the second and final per-epoch slot for that lesion.
        selected = self._repeat_positive_candidates(
            boundary, boundary_count, rng, lesion_counts
        ) + self._repeat_positive_candidates(
            complete, complete_count, rng, lesion_counts
        )
        if len(selected) < count:
            selected.extend(
                self._repeat_positive_candidates(
                    candidates,
                    count - len(selected),
                    rng,
                    lesion_counts,
                )
            )
        return selected

    def _select_positives(
        self, count: int, rng: np.random.Generator
    ) -> list[int]:
        lesion_metadata = getattr(self.dataset, "patch_request_lesions", None)
        lesion_counts: defaultdict[tuple[int, str, int], int] | None = (
            defaultdict(int) if lesion_metadata is not None else None
        )
        selected: list[int] = []
        hard_count = min(
            count,
            int(round(count * self.hard_positive_fraction)),
        )
        if hard_count and self.hard_positive_scores:
            selected.extend(
                self._repeat_positive_candidates(
                    self._rank_hard_positives_by_case(),
                    hard_count,
                    rng,
                    lesion_counts,
                    maximum_per_lesion=(
                        self.maximum_hard_positive_views_per_lesion_per_epoch
                    ),
                    preserve_order=True,
                )
            )
        ordinary_count = count - len(selected)
        size_groups = getattr(self.dataset, "positive_indices_by_scan_group", None)
        if not size_groups or not any(size_groups.values()):
            selected.extend(
                self._select_positive_subset(
                    self.dataset.positive_indices,
                    ordinary_count,
                    rng,
                    lesion_counts,
                )
            )
        else:
            small_count = int(
                round(
                    ordinary_count
                    * self.positive_scan_group_fractions["small_only"]
                )
            )
            complete_count = int(
                round(
                    ordinary_count
                    * self.positive_category_fractions["complete"]
                )
            )
            boundary_count = ordinary_count - complete_count
            small_complete_minimum = max(0, small_count - boundary_count)
            small_complete_maximum = min(small_count, complete_count)
            small_complete = min(
                small_complete_maximum,
                max(
                    small_complete_minimum,
                    int(
                        round(
                            small_count
                            * complete_count
                            / max(ordinary_count, 1)
                        )
                    ),
                ),
            )
            quotas = {
                ("small_only", "complete"): small_complete,
                ("small_only", "boundary"): small_count - small_complete,
                ("large_containing", "complete"): complete_count
                - small_complete,
                ("large_containing", "boundary"): boundary_count
                - (small_count - small_complete),
            }
            cells = list(quotas)
            if self.training:
                rng.shuffle(cells)
            categories = getattr(
                self.dataset, "positive_indices_by_category", {}
            )
            for group, category in cells:
                group_indices = set(size_groups.get(group, []))
                candidates = [
                    index
                    for index in categories.get(category, [])
                    if index in group_indices
                ]
                selected.extend(
                    self._repeat_positive_candidates(
                        candidates,
                        quotas[(group, category)],
                        rng,
                        lesion_counts,
                    )
                )
        if len(selected) < count:
            selected.extend(
                self._select_positive_subset(
                    self.dataset.positive_indices,
                    count - len(selected),
                    rng,
                    lesion_counts,
                )
            )
        if len(selected) < count:
            raise ValueError(
                "Positive lesion-view cap cannot satisfy the requested epoch size"
            )
        if self.training:
            rng.shuffle(selected)
        return selected

    def _take_unique_slots(
        self,
        candidates: Sequence[int],
        count: int,
        used_indices: set[int],
        used_slots: set[tuple[int, int, int]],
    ) -> list[int]:
        selected: list[int] = []
        for index in candidates:
            if index in used_indices:
                continue
            slot = self.dataset.patch_request_slots[index]
            if slot is None or slot in used_slots:
                continue
            selected.append(int(index))
            used_indices.add(int(index))
            used_slots.add(slot)
            if len(selected) == count:
                break
        return selected

    def _select_negatives(
        self, count: int, rng: np.random.Generator
    ) -> list[int]:
        used_indices: set[int] = set()
        used_slots: set[tuple[int, int, int]] = set()
        selected: list[int] = []
        hard_count = min(
            count,
            int(round(count * self.hard_negative_fraction)),
            len(self.hard_negative_scores),
        )
        hard_ranked = self._rank_hard_negatives_by_case()
        selected.extend(
            self._take_unique_slots(
                hard_ranked, hard_count, used_indices, used_slots
            )
        )

        ordinary_target = count - len(selected)
        category_counts = self._category_counts(ordinary_target)
        category_order = list(self.negative_categories)
        if self.training:
            rng.shuffle(category_order)
        for category in category_order:
            candidates = self._round_robin_cases(
                self.dataset.negative_indices_by_category[category], rng
            )
            selected.extend(
                self._take_unique_slots(
                    candidates,
                    category_counts[category],
                    used_indices,
                    used_slots,
                )
            )

        if len(selected) < count:
            fallback = self._round_robin_cases(self.all_negative_indices, rng)
            selected.extend(
                self._take_unique_slots(
                    fallback,
                    count - len(selected),
                    used_indices,
                    used_slots,
                )
            )
        if len(selected) < count:
            raise ValueError(
                f"Only {len(selected)} non-overlapping negative slots are available for "
                f"{count} positives; reduce positives or patch size"
            )
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        padded_count = len(self) * self.per_class
        positives = self._select_positives(padded_count, rng)
        negatives = self._select_negatives(padded_count, rng)
        if self.training:
            rng.shuffle(negatives)
        for offset in range(0, padded_count, self.per_class):
            positive_batch = positives[offset : offset + self.per_class]
            negative_batch = negatives[offset : offset + self.per_class]
            batch: list[int] = []
            for positive, negative in zip(
                positive_batch, negative_batch, strict=True
            ):
                batch.extend((positive, negative))
            if self.training:
                rng.shuffle(batch)
            yield batch
