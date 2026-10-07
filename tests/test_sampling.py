from collections import Counter

import pytest

from rnv_seg.data.sampling import BalancedPatchBatchSampler


class SyntheticPatchPool:
    def __init__(self) -> None:
        self.patch_requests = [
            (0, (0, 0, 0)),
            (1, (0, 0, 0)),
            (0, (0, 0, 32)),
            (1, (0, 0, 32)),
            *( (index % 2, (0, 0, index * 32)) for index in range(4, 16) ),
        ]
        self.positive_indices = [0, 1, 2, 3]
        self.positive_indices_by_category = {
            "complete": [0, 1],
            "boundary": [2, 3],
        }
        self.positive_indices_by_scan_group = {
            "small_only": [0, 2],
            "large_containing": [1, 3],
        }
        self.negative_indices_by_category = {
            "adjacent": [8, 9, 10, 11],
            "anatomical_coverage": [4, 5, 6, 7, 12, 13, 14, 15],
        }
        self.patch_request_slots = [None] * 4 + [
            (index % 2, index, index * 32) for index in range(4, 16)
        ]
        self.patch_request_lesions = [
            (0, "rnv", 1),
            (1, "rnv", 1),
            (0, "rnv", 2),
            (1, "rnv", 2),
            *([None] * 12),
        ]


def test_balanced_batch_sampler_enforces_exact_one_to_one_batches() -> None:
    dataset = SyntheticPatchPool()
    sampler = BalancedPatchBatchSampler(
        dataset,
        batch_size=2,
        config={
            "positive_fraction": 0.5,
            "negative_category_fractions": {
                "adjacent": 0.3,
                "anatomical_coverage": 0.7,
            },
            "hard_negative_fraction": 0.5,
        },
        seed=42,
        training=True,
    )
    sampler.set_epoch(1)
    batches = list(sampler)
    assert len(batches) == 4
    assert all(
        sum(index in dataset.positive_indices for index in batch) == 1
        for batch in batches
    )
    negative_indices = [
        index for batch in batches for index in batch if index not in dataset.positive_indices
    ]
    assert len(negative_indices) == len(set(negative_indices)) == 4
    assert sum(index in dataset.negative_indices_by_category["adjacent"] for index in negative_indices) == 1
    assert sum(index in dataset.negative_indices_by_category["anatomical_coverage"] for index in negative_indices) == 3
    positive_indices = [
        index for batch in batches for index in batch if index in dataset.positive_indices
    ]
    assert sum(index in dataset.positive_indices_by_category["complete"] for index in positive_indices) == 2
    assert sum(index in dataset.positive_indices_by_category["boundary"] for index in positive_indices) == 2


def test_balanced_sampler_allocates_exact_region_independent_negative_quotas() -> None:
    sampler = BalancedPatchBatchSampler(
        SyntheticPatchPool(),
        batch_size=2,
        config={
            "positive_fraction": 0.5,
            "negative_category_fractions": {
                "adjacent": 0.3,
                "anatomical_coverage": 0.7,
            },
        },
        seed=42,
        training=False,
    )
    assert sampler._category_counts(10) == {
        "adjacent": 3,
        "anatomical_coverage": 7,
    }


def test_positive_sampling_caps_each_lesion_to_two_views_per_epoch() -> None:
    dataset = SyntheticPatchPool()
    dataset.patch_request_lesions[:4] = [
        (0, "rnv", 1),
        (0, "rnv", 1),
        (0, "rnv", 1),
        (1, "rnv", 1),
    ]
    sampler = BalancedPatchBatchSampler(
        dataset,
        batch_size=2,
        config={
            "positive_fraction": 0.5,
            "maximum_positive_views_per_lesion_per_epoch": 2,
        },
        seed=42,
        training=True,
    )

    batches = list(sampler)
    positives = [
        index
        for batch in batches
        for index in batch
        if index in dataset.positive_indices
    ]
    counts = Counter(dataset.patch_request_lesions[index] for index in positives)

    assert len(positives) == 3
    assert max(counts.values()) <= 2


def test_balanced_batch_sampler_prioritizes_scored_hard_negatives() -> None:
    dataset = SyntheticPatchPool()
    sampler = BalancedPatchBatchSampler(
        dataset,
        batch_size=2,
        config={"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.set_hard_negative_scores({4: 0.99, 8: 0.95, 12: 0.2})
    selected = {index for batch in sampler for index in batch}
    assert {4, 8} <= selected


def test_hard_negative_pool_accumulates_and_removes_rescored_clean_patches() -> None:
    sampler = BalancedPatchBatchSampler(
        SyntheticPatchPool(),
        batch_size=2,
        config={"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.update_hard_negative_scores({4: 0.9}, [4], maximum_pool_size=2)
    sampler.update_hard_negative_scores({8: 0.8, 12: 0.7}, [8, 12], maximum_pool_size=2)
    assert sampler.hard_negative_scores == {4: 0.9, 8: 0.8}

    sampler.update_hard_negative_scores({}, [4], maximum_pool_size=2)
    assert sampler.hard_negative_scores == {8: 0.8}


def test_hard_negative_pool_ages_duplicates_and_limits_each_case() -> None:
    dataset = SyntheticPatchPool()
    dataset.patch_request_slots[8] = dataset.patch_request_slots[4]
    sampler = BalancedPatchBatchSampler(
        dataset,
        batch_size=2,
        config={"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.update_hard_negative_scores(
        {4: 0.8, 5: 0.75, 6: 0.7, 7: 0.65, 8: 0.9},
        [4, 5, 6, 7, 8],
        maximum_pool_size=10,
        maximum_per_case=1,
    )
    assert sampler.hard_negative_scores == {8: 0.9, 5: 0.75}

    sampler.update_hard_negative_scores(
        {},
        [],
        maximum_pool_size=10,
        score_decay=0.4,
        minimum_retained_score=0.35,
        maximum_per_case=1,
    )
    assert sampler.hard_negative_scores == {8: pytest.approx(0.36)}


def test_hard_negative_ranking_round_robins_cases() -> None:
    sampler = BalancedPatchBatchSampler(
        SyntheticPatchPool(),
        batch_size=2,
        config={"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.set_hard_negative_scores({4: 0.99, 6: 0.98, 5: 0.97})
    assert sampler._rank_hard_negatives_by_case()[:2] == [4, 5]


def test_hard_negative_candidates_mix_rechecks_and_new_discovery() -> None:
    sampler = BalancedPatchBatchSampler(
        SyntheticPatchPool(),
        batch_size=2,
        config={"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.set_hard_negative_scores({4: 0.9, 5: 0.8})
    candidates = sampler.hard_negative_candidates(
        limit=6,
        refresh_index=0,
        recheck_fraction=1.0 / 3.0,
    )
    assert {4, 5} <= set(candidates)
    assert len(candidates) == 6
    assert any(index not in {4, 5} for index in candidates)


def test_hard_positive_pool_prioritizes_low_confidence_views() -> None:
    dataset = SyntheticPatchPool()
    sampler = BalancedPatchBatchSampler(
        dataset,
        batch_size=2,
        config={
            "positive_fraction": 0.5,
            "hard_positive_fraction": 0.5,
            "maximum_positive_views_per_lesion_per_epoch": 1,
            "hard_positive_mining": {
                "maximum_views_per_lesion_per_epoch": 2,
            },
        },
        seed=7,
        training=False,
    )
    sampler.set_hard_positive_scores({2: 1.2, 3: 1.1})

    positives = [
        index
        for batch in sampler
        for index in batch
        if index in dataset.positive_indices
    ]

    assert {2, 3} <= set(positives)


def test_hard_positive_candidates_mix_rechecks_and_new_lesions() -> None:
    sampler = BalancedPatchBatchSampler(
        SyntheticPatchPool(),
        batch_size=2,
        config={"positive_fraction": 0.5},
        seed=7,
        training=False,
    )
    sampler.set_hard_positive_scores({0: 0.9})
    candidates = sampler.hard_positive_candidates(
        limit=3,
        refresh_index=0,
        recheck_fraction=1.0 / 3.0,
    )

    assert 0 in candidates
    assert len(candidates) == len(set(candidates)) == 3
    assert any(index != 0 for index in candidates)
