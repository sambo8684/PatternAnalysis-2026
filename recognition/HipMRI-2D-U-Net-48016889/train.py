import argparse
import csv
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import modules
from dataset import get_dataloaders, DEFAULT_ROOT, NUM_CLASSES, PROSTATE_LABEL

SMOOTH = 1e-5

# configuration 

ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
NUM_WORKERS = 4
EARLY_STOP = False 
SMOOTH = 1e-5

OUT_DIR = "/improved_train_output"
RUN_NAME = "improved"
AUG = "improved"
PROSTATE_WEIGHT = 3.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#metrics
def dice_per_class(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice per class, computed over the whole batch.

    probs, target: (B, C, H, W). Returns tensor of shape (C,).
    """
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (2 * inter + SMOOTH) / (denom + SMOOTH)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, weights=None) -> torch.Tensor:
    """1 - (optionally class-weighted) mean soft Dice over classes (softmax over channels)."""
    d = dice_per_class(F.softmax(logits, dim=1), target)
    if weights is None:
        return 1.0 - d.mean()
    return 1.0 - (d * weights).sum() / weights.sum()


def combined_loss(logits: torch.Tensor, target_onehot: torch.Tensor, weights=None) -> torch.Tensor:
    """Cross-entropy + Dice. CE gives stable gradients; Dice handles the class
    imbalance (prostate is a tiny fraction of the pixels). `weights` is an optional (C,)
    tensor of class weights used in both terms; None reproduces the unweighted loss."""
    ce = F.cross_entropy(logits, target_onehot.argmax(dim=1), weight=weights)
    return ce + dice_loss(logits, target_onehot, weights)


@torch.no_grad()
def hard_counts(logits: torch.Tensor, target: torch.Tensor):
    """Per-class sums from argmax predictions, for aggregating Dice/IoU over a whole split:
    returns (intersection, predicted pixels, target pixels), each of shape (C,)."""
    pred = F.one_hot(logits.argmax(dim=1), NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    return (pred * target).sum(dims), pred.sum(dims), target.sum(dims)


@torch.no_grad()
def evaluate(model, loader, device, weights=None):
    """Returns (mean loss, per-class Dice, per-class IoU) over a loader.
    Dice/IoU are computed over the whole split (not averaged per batch). `weights` are the
    class weights used for the reported loss only; Dice/IoU never use them."""
    model.eval()
    total_loss, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_s = torch.zeros(NUM_CLASSES, device=device)
    targ_s = torch.zeros(NUM_CLASSES, device=device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += combined_loss(logits, masks, weights).item() * images.size(0)
        n += images.size(0)
        i, p, t = hard_counts(logits, masks)
        inter += i
        pred_s += p
        targ_s += t
    dice = (2 * inter + SMOOTH) / (pred_s + targ_s + SMOOTH)
    iou = (inter + SMOOTH) / (pred_s + targ_s - inter + SMOOTH)
    return total_loss / n, dice.cpu(), iou.cpu()

def set_seed(seed: int):
    """Seed all RNGs. cuDNN autotuning is off, but GPU runs are still not guaranteed to be
    bit-for-bit identical between runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    def out(name):
        return os.path.join(OUT_DIR, f"{RUN_NAME}_{name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | run: {RUN_NAME} | aug: {AUG} | "
          f"prostate weight: {PROSTATE_WEIGHT} | seed: {SEED}")
    set_seed(SEED)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        early_stop=EARLY_STOP, aug_mode=AUG,
    )

    # n_classes must match the one-hot mask channels produced by dataset.py
    model = modules.UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optional class weights (None keeps the original unweighted loss)
    class_weights = None
    if PROSTATE_WEIGHT != 1.0:
        class_weights = torch.ones(NUM_CLASSES, device=device)
        class_weights[PROSTATE_LABEL] = PROSTATE_WEIGHT

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_prostate_dice = -1.0
    ckpt_path = out("best.pth")

    print("Starting training")
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss, seen = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)                 # (B, NUM_CLASSES, H, W)
            loss = combined_loss(logits, masks, class_weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / seen
        val_loss, val_dice, _ = evaluate(model, val_loader, device, class_weights)
        prostate = val_dice[PROSTATE_LABEL].item()
        lr = optimizer.param_groups[0]["lr"]       # learning rate used during this epoch
        scheduler.step(prostate)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "val_prostate_dice": prostate, "val_mean_dice": val_dice.mean().item(),
                        "lr": lr})
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val prostate Dice {prostate:.4f} | "
              f"val mean Dice {history[-1]['val_mean_dice']:.4f} | lr {lr:.2e}")

        # Keep the checkpoint with the best validation prostate Dice (never selected on test)
        if prostate > best_prostate_dice:
            best_prostate_dice = prostate
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best model (prostate Dice {prostate:.4f})")

    train_time = time.time() - start
    print(f"Training finished in {train_time / 60:.1f} min")

    with open(out("history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE + Dice)")
    axes[0].set_title(f"Loss ({RUN_NAME})")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_prostate_dice"] for h in history], label="prostate Dice")
    axes[1].plot(epochs, [h["val_mean_dice"] for h in history], label="mean Dice (all classes)")
    axes[1].axhline(0.75, color="grey", linestyle="--", label="target 0.75")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title(f"Validation Dice ({RUN_NAME})")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out("training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved curves to {out('training_curves.png')}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, device, class_weights)
    print(f"\nTest loss: {test_loss:.4f}")
    for c in range(NUM_CLASSES):
        tag = " (prostate)" if c == PROSTATE_LABEL else ""
        print(f"  class {c}{tag}: Dice {test_dice[c]:.4f} | IoU {test_iou[c]:.4f}")
    print(f"Test prostate Dice: {test_dice[PROSTATE_LABEL]:.4f} (target >= 0.75)")

    # Inference latency per slice (batch of 1, after warm-up)
    model.eval()
    sample = next(iter(test_loader))[0][:1].to(device)
    runs = 50
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            model(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / runs * 1000
    print(f"Inference latency: {latency_ms:.2f} ms / slice")
    print(f"Parameters: {n_params:,}")

    # Profile VRAM usage
    max_vram = None
    if torch.cuda.is_available():
        max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU VRAM Allocated: {max_vram:.2f} MB")

    with open(out("test_metrics.txt"), "w") as f:
        f.write(f"run_name {RUN_NAME}\naug {AUG}\nprostate_weight {PROSTATE_WEIGHT}\n"
                f"seed {SEED}\nepochs {EPOCHS}\n")
        f.write(f"test_loss {test_loss:.4f}\n")
        for c in range(NUM_CLASSES):
            f.write(f"class_{c} dice {test_dice[c]:.4f} iou {test_iou[c]:.4f}\n")
        f.write(f"params {n_params}\nlatency_ms {latency_ms:.2f}\n"
                f"train_minutes {train_time / 60:.1f}\n")
        f.write("peak_vram_mb " + (f"{max_vram:.2f}" if max_vram is not None else "n/a") + "\n")


if __name__ == "__main__":
    main()
