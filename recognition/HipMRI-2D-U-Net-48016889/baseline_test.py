import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# UNet Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Nifti Dataset Loader
class NiftiSlicesDataset(Dataset):
    def __init__(self, image_dir, target_dir=None, max_samples=None):
        # Find all .nii.gz files in the local directory
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, "*.nii.gz"))) if target_dir else None
        
        # Smoke-test optimization: limit samples if requested
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            if self.target_paths:
                self.target_paths = self.target_paths[:max_samples]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Nifti image using nibabel
        img_nii = nib.load(self.image_paths[idx])
        image_np = img_nii.get_fdata().astype(np.float32)
        
        # Ensure proper 2D shape [Channels, Height, Width]
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=0)
        elif image_np.ndim == 3:
            # If stored as (H, W, 1) or similar, transpose to (1, H, W)
            image_np = np.transpose(image_np, (2, 0, 1)) if image_np.shape[0] != 1 else image_np

        image = torch.tensor(image_np, dtype=torch.float32)
        
        # Normalize image intensity to [0, 1] if not already
        if image.max() > 1.0:
            image = image / image.max()

        if self.target_paths:
            target_nii = nib.load(self.target_paths[idx])
            target_np = target_nii.get_fdata().astype(np.float32)
            
            if target_np.ndim == 2:
                target_np = np.expand_dims(target_np, axis=0)
            elif target_np.ndim == 3 and target_np.shape[0] != 1:
                target_np = np.transpose(target_np, (2, 0, 1))
                
            target = torch.tensor(target_np, dtype=torch.float32)
            # Ensure binary mask values are strictly 0 or 1
            target = (target > 0.5).float()
        else:
            raise ValueError("Target segmentation masks must be provided.")

        return image, target

# ----------------------
# Data Loaders (Smoke Test Mode: max_samples=20)
# ----------------------
data_dir = "keras_slices_data"

train_dataset = NiftiSlicesDataset(
    os.path.join(data_dir, "keras_slices_train"),
    target_dir=os.path.join(data_dir, "keras_slices_seg_train"),
    max_samples=20  # Limits to 20 slices for quick smoke test validation
)

val_dataset = NiftiSlicesDataset(
    os.path.join(data_dir, "keras_slices_validate"),
    target_dir=os.path.join(data_dir, "keras_slices_seg_validate"),
    max_samples=10
)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# Training Setup (Smoke Test)
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet(n_channels=1, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 3

print("Starting smoke test training loop...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# Profile VRAM usage
if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU VRAM Allocated: {max_vram:.2f} MB")

# ----------------------
# Visualization
# ----------------------
model.eval()
with torch.no_grad():
    images, targets = next(iter(val_loader))
    images = images.to(device)
    outputs = torch.sigmoid(model(images))
    preds = (outputs > 0.5).float()
    
    n_show = min(4, len(images))
    plt.figure(figsize=(12, 8))
    for i in range(n_show):
        plt.subplot(3, n_show, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Input MRI")
        
        plt.subplot(3, n_show, i+1+n_show)
        plt.imshow(preds[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Prediction")
        
        plt.subplot(3, n_show, i+1+2*n_show)
        plt.imshow(targets[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")
        
    plt.tight_layout()
    plt.show()

print("Smoke test complete.")