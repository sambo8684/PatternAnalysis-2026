from dataset import get_dataloaders, NUM_CLASSES, PROSTATE_LABEL
import modules
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader, test_loader = get_dataloaders(
    root="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
    batch_size=16,
    num_workers=4,
)

model = modules.UNet(n_channels=1, n_classes=1).to(device)
num_epochs = 3

print("Starting training")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

# Profile VRAM usage
if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU VRAM Allocated: {max_vram:.2f} MB")
