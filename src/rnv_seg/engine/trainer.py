from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from rnv_seg.engine.evaluator import evaluate_loader
from rnv_seg.utils.checkpoint import save_checkpoint
from rnv_seg.utils.experiment import append_jsonl


def train(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    run_dir: str | Path,
    device: torch.device,
) -> dict[str, float]:
    training = config["training"]
    epochs = int(training["epochs"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metric_name = training.get("checkpoint_metric", "dice")
    mode = training.get("checkpoint_mode", "max")
    best = -math.inf if mode == "max" else math.inf
    patience = int(training.get("early_stopping_patience", epochs))
    without_improvement = 0
    run_dir = Path(run_dir)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image, target)
                loss = criterion(output.logits, target)
                loss = loss + sum(output.auxiliary_losses.values())
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip = float(training.get("gradient_clip_norm", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
        if scheduler is not None:
            scheduler.step()
        validation_metrics = evaluate_loader(model, validation_loader, device, config)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation_metrics,
        }
        append_jsonl(run_dir / "logs" / "metrics.jsonl", row)
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, validation_metrics, config
        )
        score = validation_metrics.get(metric_name)
        improved = score is not None and ((mode == "max" and score > best) or (mode == "min" and score < best))
        if improved:
            best = float(score)
            without_improvement = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
            )
        else:
            without_improvement += 1
        if without_improvement >= patience:
            break
    return {"best_epoch_metric": best, "epochs_completed": float(epoch)}

