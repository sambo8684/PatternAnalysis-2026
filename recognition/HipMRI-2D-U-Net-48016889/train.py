from dataset import get_dataloaders, NUM_CLASSES, PROSTATE_LABEL
import modules
import torch
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

# configuration 

ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
CKPT_PATH = "best_unet.pth"
CURVES_PATH = "training_curves.png"
SMOOTH = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#metrics
def dice_per_class(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice per class, averaged over the batch.

    probs, target: (B, C, H, W). Returns tensor of shape (C,).
    """
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (2 * inter + SMOOTH) / (denom + SMOOTH)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean soft Dice over classes (softmax over channel dim)."""
    return 1.0 - dice_per_class(F.softmax(logits, dim=1), target).mean()


def combined_loss(logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + Dice. CE gives stable gradients; Dice handles the class
    imbalance (prostate is a tiny fraction of the pixels)."""
    ce = F.cross_entropy(logits, target_onehot.argmax(dim=1))
    return ce + dice_loss(logits, target_onehot)


@torch.no_grad()
def hard_dice_per_class(logits: torch.Tensor, target: torch.Tensor):
    """Dice and IoU per class from argmax predictions. Returns sums needed to
    aggregate over the whole dataset: (intersection, pred_sum, target_sum)."""
    pred = F.one_hot(logits.argmax(dim=1), NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    return (pred * target).sum(dims), pred.sum(dims), target.sum(dims)


@torch.no_grad()
def evaluate(model, loader):
    """Returns (mean loss, per-class Dice, per-class IoU) over a loader.
    Dice/IoU are computed over the whole split (not averaged per batch)."""
    model.eval()
    total_loss, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_s = torch.zeros(NUM_CLASSES, device=device)
    targ_s = torch.zeros(NUM_CLASSES, device=device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += combined_loss(logits, masks).item() * images.size(0)
        n += images.size(0)
        i, p, t = hard_dice_per_class(logits, masks)
        inter += i
        pred_s += p
        targ_s += t
    dice = (2 * inter + SMOOTH) / (pred_s + targ_s + SMOOTH)
    iou = (inter + SMOOTH) / (pred_s + targ_s - inter + SMOOTH)
    return total_loss / n, dice.cpu(), iou.cpu()

# load data/model
train_loader, val_loader, test_loader = get_dataloaders(
    root=ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)

model = modules.UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

# training loop
train_losses, val_losses, val_prostate_dice, val_mean_dice = [], [], [], []
best_prostate_dice = -1.0

print("Starting training")
start = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, seen = 0.0, 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        seen += images.size(0)

    train_loss = running_loss / seen
    val_loss, val_dice, _ = evaluate(model, val_loader)
    prostate = val_dice[PROSTATE_LABEL].item()
    scheduler.step(prostate)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_prostate_dice.append(prostate)
    val_mean_dice.append(val_dice.mean().item())

    print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | train loss {train_loss:.4f} | "
          f"val loss {val_loss:.4f} | val prostate Dice {prostate:.4f} | "
          f"val mean Dice {val_mean_dice[-1]:.4f}")

    if prostate > best_prostate_dice:
        best_prostate_dice = prostate
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  -> saved new best model (prostate Dice {prostate:.4f})")

train_time = time.time() - start
print(f"Training finished in {train_time / 60:.1f} min")

# plotting
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs, train_losses, label="train")
axes[0].plot(epochs, val_losses, label="validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (CE + Dice)")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(epochs, val_prostate_dice, label="prostate Dice")
axes[1].plot(epochs, val_mean_dice, label="mean Dice (all classes)")
axes[1].axhline(0.75, color="grey", linestyle="--", label="target 0.75")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice")
axes[1].set_title("Validation Dice")
axes[1].legend()
plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=150)
print(f"Saved curves to {CURVES_PATH}")

# final test
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
test_loss, test_dice, test_iou = evaluate(model, test_loader)
print(f"\nTest loss: {test_loss:.4f}")
for c in range(NUM_CLASSES):
    tag = " (prostate)" if c == PROSTATE_LABEL else ""
    print(f"  class {c}{tag}: Dice {test_dice[c]:.4f} | IoU {test_iou[c]:.4f}")
print(f"Test prostate Dice: {test_dice[PROSTATE_LABEL]:.4f} (target >= 0.75)")

# calculate latency for each slice
model.eval()
sample = next(iter(test_loader))[0][:1].to(device)
with torch.no_grad():
    for _ in range(5):
        model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    runs = 50
    for _ in range(runs):
        model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
latency_ms = (time.time() - t0) / runs * 1000
print(f"Inference latency: {latency_ms:.2f} ms / slice")
print(f"Parameters: {n_params:,}")

# Profile VRAM usage
if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU VRAM Allocated: {max_vram:.2f} MB")

# write results to file
with open("test_metrics.txt", "w") as f:
    f.write(f"test_loss {test_loss:.4f}\n")
    for c in range(NUM_CLASSES):
        f.write(f"class_{c} dice {test_dice[c]:.4f} iou {test_iou[c]:.4f}\n")
    f.write(f"params {n_params}\nlatency_ms {latency_ms:.2f}\ntrain_minutes {train_time / 60:.1f}\n")