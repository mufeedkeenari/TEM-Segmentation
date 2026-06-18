import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules from the previous steps
from src.data.dataset import LBNLTEMDataset
from src.data.transforms import get_train_transforms
from src.models.unet import UNet
from src.engine.loss import BCEDiceLoss

# --- HYPERPARAMETERS ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10  # Keeping this low for rapid prototyping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model() -> None:
    print(f"--- Starting TEM Segmentation Training on {DEVICE.upper()} ---")

    # 1. Initialize the Data Pipeline
    print("Loading data engine...")
    train_transforms = get_train_transforms(crop_size=256)
    train_dataset = LBNLTEMDataset(
        images_path="data/raw/au_10nm_images.h5",
        masks_path="data/raw/au_10nm_labels.h5",
        transform=train_transforms
    )

    # DataLoader handles the multiprocessing and batching automatically
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 2. Initialize the Model, Loss, and Optimizer
    print("Initializing U-Net Architecture...")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = BCEDiceLoss()
    # Adam is the industry standard optimizer for computer vision
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. The Core Training Loop
    print(f"Beginning training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()  # Put model in training mode
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move physical data to the GPU (or CPU)
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            # Step A: Forward Pass (Make a prediction)
            predictions = model(images)

            # Step B: Calculate the Error (Dice Loss)
            loss = criterion(predictions, masks)

            # Step C: Backward Pass (Update the weights via calculus)
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Calculate new gradients
            optimizer.step()  # Update the U-Net's memory

            epoch_loss += loss.item()

        # Print metrics at the end of each epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Average Training Loss: {avg_loss:.4f}")

    # 4. Save the Model Weights
    print("--- Training Complete ---")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    save_path = "data/processed/unet_tem_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights successfully saved to: {save_path}")


if __name__ == "__main__":
    train_model()