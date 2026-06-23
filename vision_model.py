import os
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0

# Select hardware accelerator: use GPU (CUDA) if available, otherwise fallback to CPU
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class PendulumDataset(Dataset):
    """
    Custom Dataset class to load, cache, and preprocess pendulum data from HDF5.
    Resizes images to fit standard neural network inputs and applies normalization.
    """

    def __init__(self, h5_path):
        print("Loading and preprocessing dataset into RAM...")
        # Load raw data matrix slices from the HDF5 file binary structure
        with h5py.File(h5_path, "r") as f:
            raw_images = f["images"][:]
            self.coords = torch.from_numpy(f["coords"][:]).float()
            self.angles = torch.from_numpy(f["angles"][:]).float()

        # Resize all spatial matrices from simulation size down to 224x224 pixels
        self.images = []
        for img in raw_images:
            res = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            self.images.append(res)

        self.images = np.array(self.images)
        print(f"Dataset ready in RAM. Shape: {self.images.shape}")

        # Set up a transformation pipeline: Tensor conversion, 3-channel expansion, ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        # Returns the total volume of samples present inside this collection
        return len(self.images)

    def __getitem__(self, idx):
        # Apply data transformations to the grayscale image array
        image = self.transform(self.images[idx])
        
        c = self.coords[idx]
        a = self.angles[idx]
        # Target representation vector: 8 coordinates + 3 link cosines + 3 link sines = 14 values total
        target = torch.cat([c, torch.cos(a), torch.sin(a)])
        
        return image, target


def eval_epoch(model, dataloader, criterion):
    """
    Evaluates model state tracking capabilities over the target validation dataset partition.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        # Draw a single training sample batch to print model predictions against targets
        sample_img, sample_target = next(iter(train_loader))
        output = model(sample_img.to(device))
        print("Example Prediction:", output[0].cpu().numpy())
        print("Example Target:", sample_target[0].numpy())
        
        # Iterate over validation dataloader batches to accumulate validation loss metric
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    start_epoch=0,
):
    """
    Core model optimization execution wrapper managing state updates, logs, and checkpoints.
    """
    best_val_loss = float("inf")
    best_weight_path = "best_model.pth"
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nStarting Epoch {epoch} | LR: {current_lr:.6f}")
        
        # Batch gradient descent parameter adjustment iteration steps
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Reset gradients, calculate forward pass loss, compute backpropagation
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Clip gradient norms to stabilize parameters against numerical explosion forces
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        # Metric consolidation phase at the end of each training epoch loop
        epoch_loss = running_loss / len(train_loader.dataset)
        eval_loss = eval_epoch(model, val_loader, criterion)
        
        # Update scheduler state according to validation loss criteria performance
        scheduler.step(eval_loss)
        
        final_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.6f}, "
            f"Val Loss: {eval_loss:.6f}, LR: {final_lr:.6f}"
        )

        # Persistent storage checkpoint mechanism tracking best-performing configurations
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), best_weight_path)
            print(f"--- Model Saved (New Best Val Loss: {best_val_loss:.6f}) ---")


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    INITIAL_LR = 1e-4

    print(f"Using device: {device}")
    
    best_weight_path = "best_model.pth"

    # Instantiate a ShuffleNetV2 topology and rebuild its final linear head projection array
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, 14),
    )

    # Resume prior training sequence or begin initialization procedures
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        print("Weights loaded. Continuing training...")
    else:
        print("Starting training from scratch.")

    model.to(device)

    # Enable parameter optimization updates across all graph variables
    for param in model.parameters():
        param.requires_grad = True
    
    # Configure optimizer and robust regression loss criterion algorithms
    optimizer = optim.AdamW(
        model.parameters(), lr=INITIAL_LR, weight_decay=1e-2
    )
    criterion = nn.SmoothL1Loss(beta=0.1)

    # Dataset partition splitting initialization (80% training set, 20% validation set)
    dataset = PendulumDataset("triple_pendulum_dataset.hdf5")
    train_size = int(0.8 * len(dataset))
    train, val = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    # Construct continuous iterable memory-pinned batch management stream loaders
    train_loader = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    # Dynamic adjustment scheduler setup to lower learning rates upon validation stagnation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=False,
    )

    # Launch model optimization sequences
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=NUM_EPOCHS,
    )