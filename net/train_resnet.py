import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.amp import GradScaler, autocast

import net.res_u_net as net
from sklearn.metrics import jaccard_score


# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths  # List of image file paths
        self.mask_paths = mask_paths    # List of mask file paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])  # Assumes mask is single-channel (grayscale)
        mask = np.array(mask)  # Convert to numpy for processing
        mask = torch.tensor(mask, dtype=torch.long)  # Shape: (1024, 1024)

        if self.transform:
            image = self.transform(image)

        return image, mask

# Data preparation
image_paths = ["path/to/images/1.jpg", "path/to/images/2.jpg", ...]  # Replace with your paths
mask_paths = ["path/to/masks/1.png", "path/to/masks/2.png", ...]
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)  # Small batch due to memory

model = net.get_model()
model = model.cuda() if torch.cuda.is_available() else model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # For mixed precision
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Optional LR decay

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.cuda() if torch.cuda.is_available() else images  # Shape: (batch, 3, 1024, 1024)
        masks = masks.cuda() if torch.cuda.is_available() else masks    # Shape: (batch, 1024, 1024)

        optimizer.zero_grad()
        with autocast():  # Mixed precision
            outputs = model(images)  # Shape: (batch, num_classes, 1024, 1024)
            loss = criterion(outputs, masks)  # Compute loss

        scaler.scale(loss).backward()  # Scale gradients
        scaler.step(optimizer)         # Update weights
        scaler.update()
        running_loss += loss.item()

    scheduler.step()  # Update learning rate
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"checkpoint_epoch_{epoch+1}.pth")

# Validation (optional, requires validation dataloader)
# Compute metrics like IoU

def compute_iou(outputs, masks, num_classes):
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: (batch, 1024, 1024)
    masks = masks.cpu().numpy()
    iou = [jaccard_score(masks.flatten(), outputs.flatten(), average=None, labels=[i]) for i in range(num_classes)]
    return np.mean(iou)

# Example validation
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# model.eval()
# with torch.no_grad():
#     val_iou = 0.0
#     for images, masks in val_dataloader:
#         images, masks = images.cuda(), masks.cuda()
#         outputs = model(images)
#         val_iou += compute_iou(outputs, masks, num_classes)
#     print(f"Validation IoU: {val_iou/len(val_dataloader):.4f}")