import os
import re
from glob import glob
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import nifti

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
NUM_CLASSES = 6
PROSTATE_LABEL = 5
EXPECTED_SHAPE = (256, 128)

_SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# Exclude minority image shape from the dataset
def _list_nifti(folder: str):
    files = sorted(glob(os.path.join(folder, "*.nii.gz")) + glob(os.path.join(folder, "*.nii")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {folder}")
    return files


_KEY_RE = re.compile(r"(\d+_week_\d+_slice_\d+)")
_CASE_RE = re.compile(r"(\d+)_week_\d+_slice_\d+")


def _pair_key(path: str) -> str:
    """Key identifying a slice, ignoring the image/label filename prefix."""
    m = _KEY_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def _case_id(path: str):
    """Patient (case) ID from a filename such as case_004_week_0_slice_12, else None."""
    m = _CASE_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def _filter_by_shape(img_paths, seg_paths, expected=EXPECTED_SHAPE):
    """Keep only image/mask pairs where BOTH have the expected 2D shape.
    Only the NIfTI header is read, so this is fast."""
    keep_imgs, keep_segs = [], []
    for ip, sp in zip(img_paths, seg_paths):
        if nib.load(ip).shape[:2] == expected and nib.load(sp).shape[:2] == expected:
            keep_imgs.append(ip)
            keep_segs.append(sp)
    dropped = len(img_paths) - len(keep_imgs)
    if dropped:
        print(f"  excluded {dropped} slices with shape != {expected}")
    return keep_imgs, keep_segs


def get_split_paths(root: str, split: str):
    """Return aligned (image_paths, mask_paths) for a split (non-standard shapes removed)."""
    if split not in _SPLIT_DIRS:
        raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got '{split}'")
    img_dir, seg_dir = (os.path.join(root, d) for d in _SPLIT_DIRS[split])
    img_paths, seg_paths = _list_nifti(img_dir), _list_nifti(seg_dir)

    if len(img_paths) != len(seg_paths):
        raise ValueError(f"[{split}] {len(img_paths)} images vs {len(seg_paths)} masks")

    # Match by slice key so image/mask misalignment is caught, not silently trained on
    seg_by_key = {_pair_key(p): p for p in seg_paths}
    aligned_segs = []
    for p in img_paths:
        key = _pair_key(p)
        if key not in seg_by_key:
            raise ValueError(f"[{split}] no mask found for image {os.path.basename(p)}")
        aligned_segs.append(seg_by_key[key])
    return _filter_by_shape(img_paths, aligned_segs)


def get_case_ids(root: str, split: str):
    """Set of patient IDs found in a split's image filenames."""
    img_dir = os.path.join(root, _SPLIT_DIRS[split][0])
    return {c for c in (_case_id(p) for p in _list_nifti(img_dir)) if c is not None}


def check_patient_leakage(root: str = DEFAULT_ROOT, verbose: bool = True):
    """Raise ValueError if any patient ID occurs in more than one split.

    Uses filenames only (cheap). If IDs cannot be parsed from the filenames the check
    cannot run; a warning is printed instead.
    """
    ids = {s: get_case_ids(root, s) for s in _SPLIT_DIRS}
    if any(len(v) == 0 for v in ids.values()):
        print("WARNING: could not parse patient IDs from filenames; leakage check skipped.")
        return ids
    names = list(ids)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = ids[a] & ids[b]
            if common:
                overlaps[(a, b)] = sorted(common)
    if overlaps:
        msg = "; ".join(f"{a}/{b}: {c}" for (a, b), c in overlaps.items())
        raise ValueError(f"Patient leakage between splits -> {msg}")
    if verbose:
        print("Patient leakage check passed: " +
              ", ".join(f"{s} {len(v)} patients" for s, v in ids.items()))
    return ids


class HipMRIDataset(Dataset):
    """HipMRI 2D slices.

    Each item is (image, mask):
        image: float32 tensor (1, H, W), z-score normalised per slice
        mask:  float32 one-hot tensor (NUM_CLASSES, H, W)
    Set `return_labels=True` to get an int64 label map (H, W) instead of one-hot
    (useful for CrossEntropyLoss / metric computation).
    """

    def __init__(self, root: str = DEFAULT_ROOT, split: str = "train",
                 num_classes: int = NUM_CLASSES, augment: bool = False,
                 return_labels: bool = False, early_stop: bool = False,
                 aug_mode: str = "baseline"):
        if aug_mode not in ("baseline", "improved"):
            raise ValueError(f"aug_mode must be 'baseline' or 'improved', got '{aug_mode}'")
        self.split = split
        self.num_classes = num_classes
        self.augment = augment
        self.aug_mode = aug_mode
        self.return_labels = return_labels

        img_paths, seg_paths = get_split_paths(root, split)
        self.image_paths = img_paths
        self.mask_paths = seg_paths

        images = nifti.load_data_2D(img_paths, normImage=True, early_stop=early_stop)
        # Labels are small integers, so load/store them as uint8 (8x less RAM than int64);
        # they are converted to int64 per item in __getitem__ where torch needs it.
        masks = nifti.load_data_2D(seg_paths, normImage=False, dtype=np.uint8,
                             early_stop=early_stop)

        self.images = torch.from_numpy(images).unsqueeze(1)  # (N,1,H,W) float32
        self.labels = torch.from_numpy(masks)                # (N,H,W) uint8

        if int(self.labels.max()) >= num_classes:
            raise ValueError(
                f"[{split}] found label {int(self.labels.max())} but num_classes={num_classes}")

    def __len__(self):
        return self.images.shape[0]

    def _augment(self, image: torch.Tensor, label: torch.Tensor):
        """Augmentation, see the module docstring for the two modes. Flips are applied to
        image and label together; intensity jitter to the image only."""
        if torch.rand(1).item() < 0.5:                # left-right flip
            image, label = image.flip(-1), label.flip(-1)
        if self.aug_mode == "baseline":
            if torch.rand(1).item() < 0.5:            # vertical flip (original pipeline)
                image, label = image.flip(-2), label.flip(-2)
        else:
            scale = 0.9 + 0.2 * torch.rand(1).item()      # contrast 0.9 - 1.1
            shift = (torch.rand(1).item() - 0.5) * 0.2    # brightness +-0.1 (z-score units)
            image = image * scale + shift
        return image, label

    def __getitem__(self, idx: int):
        image, label = self.images[idx], self.labels[idx].long()
        if self.augment:
            image, label = self._augment(image, label)
        if self.return_labels:
            return image, label
        onehot = F.one_hot(label, self.num_classes).permute(2, 0, 1).float()
        return image, onehot



def get_dataloaders(root: str = DEFAULT_ROOT, batch_size: int = 16, num_workers: int = 2,
                    augment_train: bool = True, return_labels: bool = False,
                    early_stop: bool = False, check_leakage: bool = True,
                    aug_mode: str = "baseline") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validate / test dataloaders from the pre-split folders."""
    if check_leakage:
        check_patient_leakage(root)
    train_ds = HipMRIDataset(root, "train", augment=augment_train,
                             return_labels=return_labels, early_stop=early_stop,
                             aug_mode=aug_mode)
    val_ds = HipMRIDataset(root, "validate", return_labels=return_labels, early_stop=early_stop)
    test_ds = HipMRIDataset(root, "test", return_labels=return_labels, early_stop=early_stop)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test: python dataset.py [root]
    import sys
    r = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    check_patient_leakage(r)
    ds = HipMRIDataset(r, "validate", early_stop=True)
    x, y = ds[0]
    print(f"slices: {len(ds)}  image: {tuple(x.shape)}  mask: {tuple(y.shape)}")
    print("classes present in first 22 slices:", torch.unique(ds.labels[:22]).tolist())