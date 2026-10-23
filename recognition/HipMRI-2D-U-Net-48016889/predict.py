import argparse
import csv
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import modules
from dataset import HipMRIDataset, DEFAULT_ROOT, NUM_CLASSES, PROSTATE_LABEL

BLADDER_LABEL, RECTUM_LABEL = 3, 4
SMOOTH = 1e-5
_NAME_RE = re.compile(r"(\d+)_week_(\d+)_slice_(\d+)")

def parse_name(path: str, fallback_idx: int):
    """Return (case, week, slice_idx) from a filename; fall back to the order index."""
    m = _NAME_RE.search(os.path.basename(path))
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return "unknown", "0", fallback_idx


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    """Dice and IoU between two boolean masks."""
    inter = np.logical_and(pred, gt).sum()
    p, g = pred.sum(), gt.sum()
    return (2 * inter + SMOOTH) / (p + g + SMOOTH), (inter + SMOOTH) / (p + g - inter + SMOOTH)


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=32):
    """Argmax label map (N, H, W) uint8 for every slice in the dataset."""
    model.eval()
    preds = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.images[start:start + batch_size].to(device)
        preds.append(model(batch).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


def overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """RGB overlay on the MR slice: green = true positive, red = false positive
    (over-contouring), blue = false negative (under-contouring)."""
    lo, hi = image.min(), image.max()
    gray = (image - lo) / (hi - lo + 1e-8)
    rgb = np.stack([gray] * 3, axis=-1)
    colours = {
        "tp": (np.logical_and(pred_mask, gt_mask), (0.0, 1.0, 0.0)),
        "fp": (np.logical_and(pred_mask, ~gt_mask), (1.0, 0.0, 0.0)),
        "fn": (np.logical_and(~pred_mask, gt_mask), (0.0, 0.3, 1.0)),
    }
    for mask, colour in colours.values():
        rgb[mask] = 0.4 * rgb[mask] + 0.6 * np.array(colour)
    return rgb

def per_slice_table(dataset, preds):
    """One row per test slice with prostate Dice, error pixels and boundary leakage."""
    labels = dataset.labels.numpy()
    rows = []
    for i in range(len(dataset)):
        case, week, sl = parse_name(dataset.image_paths[i], i)
        gt = labels[i] == PROSTATE_LABEL
        pr = preds[i] == PROSTATE_LABEL
        d, iou = dice_iou(pr, gt)
        rows.append({
            "idx": i, "case": case, "week": week, "slice": sl,
            "gt_px": int(gt.sum()), "pred_px": int(pr.sum()),
            "dice": float(d) if gt.any() else float("nan"),   # undefined if no prostate
            "iou": float(iou) if gt.any() else float("nan"),
            "fp_px": int(np.logical_and(pr, ~gt).sum()),
            "fn_px": int(np.logical_and(~pr, gt).sum()),
            # predicted prostate pixels that sit on ground-truth bladder / rectum
            "leak_bladder_px": int(np.logical_and(pr, labels[i] == BLADDER_LABEL).sum()),
            "leak_rectum_px": int(np.logical_and(pr, labels[i] == RECTUM_LABEL).sum()),
        })
    # relative position of each slice within its own case/week volume (0..1)
    groups = {}
    for r in rows:
        groups.setdefault((r["case"], r["week"]), []).append(r["slice"])
    for r in rows:
        s = groups[(r["case"], r["week"])]
        lo, hi = min(s), max(s)
        r["rel_pos"] = (r["slice"] - lo) / (hi - lo) if hi > lo else 0.5
    return rows


def print_overall(dataset, preds):
    labels = dataset.labels.numpy()
    print("\nOverall test results (pooled over all slices):")
    for c in range(NUM_CLASSES):
        d, iou = dice_iou(preds == c, labels == c)
        tag = " (prostate)" if c == PROSTATE_LABEL else ""
        print(f"  class {c}{tag}: Dice {d:.4f} | IoU {iou:.4f}")


def print_position_summary(rows):
    """Mean prostate Dice for thirds of the volume (only slices containing prostate)."""
    print("\nProstate Dice by position in volume (slices that contain prostate):")
    for name, lo, hi in [("low-index third", 0.0, 1 / 3), ("middle third", 1 / 3, 2 / 3),
                         ("high-index third", 2 / 3, 1.0001)]:
        sel = [r["dice"] for r in rows
               if lo <= r["rel_pos"] < hi and r["gt_px"] > 0]
        if sel:
            print(f"  {name:17s}: mean Dice {np.mean(sel):.3f}  (n={len(sel)})")
    empty = [r for r in rows if r["gt_px"] == 0]
    fp_empty = sum(1 for r in empty if r["pred_px"] > 0)
    print(f"\nSlices with no prostate in ground truth: {len(empty)}; "
          f"model predicted prostate on {fp_empty} of them")
    print(f"Total predicted-prostate pixels on GT bladder: "
          f"{sum(r['leak_bladder_px'] for r in rows)}, on GT rectum: "
          f"{sum(r['leak_rectum_px'] for r in rows)}")

def plot_per_slice(rows, path):
    has = [r for r in rows if r["gt_px"] > 0]
    pos = np.array([r["rel_pos"] for r in has])
    dsc = np.array([r["dice"] for r in has])

    bins = np.linspace(0, 1, 11)
    centres, means = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = dsc[(pos >= a) & (pos < b + (1e-9 if b == 1 else 0))]
        if len(sel):
            centres.append((a + b) / 2)
            means.append(sel.mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pos, dsc, s=8, alpha=0.35, label="individual slice")
    ax.plot(centres, means, "o-", color="C1", label="mean per position bin")
    ax.axhline(0.75, color="grey", linestyle="--", label="target 0.75")
    ax.set_xlabel("Relative slice position within volume (0 = first slice, 1 = last)")
    ax.set_ylabel("Prostate Dice")
    ax.set_title("Prostate Dice vs. slice position (test set)")
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _row_panels(axes, dataset, preds, idx):
    img = dataset.images[idx, 0].numpy()
    gt = dataset.labels[idx].numpy() == PROSTATE_LABEL
    pr = preds[idx] == PROSTATE_LABEL
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(gt, cmap="gray")
    axes[2].imshow(pr, cmap="gray")
    axes[3].imshow(overlay(img, pr, gt))
    for a in axes:
        a.axis("off")


def plot_cases(dataset, preds, rows, indices, path, title):
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.8 * n), squeeze=False)
    for k, idx in enumerate(indices):
        _row_panels(axes[k], dataset, preds, idx)
        r = rows[idx]
        axes[k, 0].set_title(f"{r['case']} wk{r['week']} slice {r['slice']}\n"
                             f"Dice {r['dice']:.2f}" if r["gt_px"] else
                             f"{r['case']} wk{r['week']} slice {r['slice']}\n(no prostate in GT)",
                             fontsize=8)
        if k == 0:
            axes[k, 1].set_title("Ground truth\n(prostate)", fontsize=8)
            axes[k, 2].set_title("Prediction\n(prostate)", fontsize=8)
            axes[k, 3].set_title("Green=TP  Red=FP  Blue=FN", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--ckpt", default="best_unet.pth")
    parser.add_argument("--out-dir", default="predict_outputs")
    parser.add_argument("--n-failures", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HipMRIDataset(args.root, "test")
    model = modules.UNet(n_channels=1, n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    preds = predict_all(model, dataset, device)
    rows = per_slice_table(dataset, preds)

    print_overall(dataset, preds)
    print_position_summary(rows)

    # per-slice CSV
    csv_path = os.path.join(args.out_dir, "per_slice_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_per_slice(rows, os.path.join(args.out_dir, "per_slice_dice.png"))

    # random examples (fixed seed, not cherry-picked)
    rng = np.random.default_rng(args.seed)
    ex = rng.choice(len(dataset), size=min(args.n_examples, len(dataset)), replace=False)
    plot_cases(dataset, preds, rows, list(ex),
               os.path.join(args.out_dir, "example_predictions.png"),
               "Random test slices (uncurated)")

    # worst failures among slices that contain prostate
    has = [r for r in rows if r["gt_px"] > 0]
    worst = sorted(has, key=lambda r: r["dice"])[:args.n_failures]
    plot_cases(dataset, preds, rows, [r["idx"] for r in worst],
               os.path.join(args.out_dir, "failure_cases.png"),
               f"{len(worst)} lowest-Dice test slices")

    print(f"\n{len(worst)} worst slices (prostate present):")
    for r in worst:
        print(f"  {r['case']} wk{r['week']} slice {r['slice']:>3d} | Dice {r['dice']:.3f} | "
              f"GT {r['gt_px']:>5d}px  pred {r['pred_px']:>5d}px | "
              f"FP {r['fp_px']:>5d}  FN {r['fn_px']:>5d}")
    print(f"\nSaved figures and CSV to {args.out_dir}/")


if __name__ == "__main__":
    main()