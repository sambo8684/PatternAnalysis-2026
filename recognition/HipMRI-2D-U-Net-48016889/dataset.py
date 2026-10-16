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

_SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

def _list_nifti(folder: str):
    files = sorted(glob(os.path.join(folder, "*.nii.gz")) + glob(os.path.join(folder, "*.nii")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {folder}")
    return files


_KEY_RE = re.compile(r"(\d+_week_\d+_slice_\d+)")

def _pair_key(path: str) -> str:
    """Key identifying a slice, ignoring the image/label filename prefix."""
    m = _KEY_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def get_split_paths(root: str, split: str):
    """Return aligned (image_paths, mask_paths) for a split."""
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
    return img_paths, aligned_segs


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
                 return_labels: bool = False, early_stop: bool = False):
        self.split = split
        self.num_classes = num_classes
        self.augment = augment
        self.return_labels = return_labels

        img_paths, seg_paths = get_split_paths(root, split)
        self.image_paths = img_paths
        self.mask_paths = seg_paths

        images = nifti.load_data_2D(img_paths, normImage=True, early_stop=early_stop)
        masks = nifti.load_data_2D(seg_paths, normImage=False, early_stop=early_stop)

        self.images = torch.from_numpy(images).unsqueeze(1)           # (N,1,H,W)
        self.labels = torch.from_numpy(np.rint(masks).astype(np.int64))  # (N,H,W)

        if int(self.labels.max()) >= num_classes:
            raise ValueError(
                f"[{split}] found label {int(self.labels.max())} but num_classes={num_classes}")

    def __len__(self):
        return self.images.shape[0]

    def _augment(self, image: torch.Tensor, label: torch.Tensor):
        """Geometric augmentation applied identically to image and label."""
        if torch.rand(1).item() < 0.5:      # horizontal flip
            image, label = image.flip(-1), label.flip(-1)
        if torch.rand(1).item() < 0.5:      # vertical flip
            image, label = image.flip(-2), label.flip(-2)
        return image, label

    def __getitem__(self, idx: int):
        image, label = self.images[idx], self.labels[idx]
        if self.augment:
            image, label = self._augment(image, label)
        if self.return_labels:
            return image, label
        onehot = F.one_hot(label, self.num_classes).permute(2, 0, 1).float()
        return image, onehot


def get_dataloaders(root: str = DEFAULT_ROOT, batch_size: int = 16, num_workers: int = 2,
                    augment_train: bool = True, return_labels: bool = False,
                    early_stop: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validate / test dataloaders from the pre-split folders."""
    train_ds = HipMRIDataset(root, "train", augment=augment_train,
                             return_labels=return_labels, early_stop=early_stop)
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
    # quick smoke test 
    import sys
    r = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    ds = HipMRIDataset(r, "validate", early_stop=True)
    x, y = ds[0]
    print(f"slices: {len(ds)}  image: {tuple(x.shape)}  mask: {tuple(y.shape)}")
    print("classes present in first 22 slices:", torch.unique(ds.labels[:22]).tolist())