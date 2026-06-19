import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.config import RAW_DATA_DIR, PROCESSED_DIR, IMAGES_FILENAME, MASKS_FILENAME, WEIGHTS_FILENAME
from src.data.dataset import LBNLTEMDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import UNet
from src.engine.loss import BCEDiceLoss

# --- HYPERPARAMETERS ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 100  # Change this to 2 for the local smoke test!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model() -> None:
    print(f"--- Starting TEM Segmentation Training on {DEVICE.upper()} ---")

    # 1. Initialize the Data Pipeline
    print("Loading data engine...")
    train_transforms = get_train_transforms(crop_size=256)
    val_transforms = get_val_transforms(crop_size=256)  # Deterministic for validation

    # Create the full dataset first to get the total length
    full_dataset = LBNLTEMDataset(
        images_path=RAW_DATA_DIR / IMAGES_FILENAME,
        masks_path=RAW_DATA_DIR / MASKS_FILENAME,
        transform=train_transforms
    )

    # --- THE VALIDATION SPLIT ---
    total_samples = len(full_dataset)
    split_idx = int(total_samples * 0.80)

    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, total_samples))

    print(f"Total samples: {total_samples} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    # Create the split datasets
    train_dataset = LBNLTEMDataset(
        images_path=RAW_DATA_DIR / IMAGES_FILENAME,
        masks_path=RAW_DATA_DIR / MASKS_FILENAME,
        transform=train_transforms,
        indices=train_indices
    )

    val_dataset = LBNLTEMDataset(
        images_path=RAW_DATA_DIR / IMAGES_FILENAME,
        masks_path=RAW_DATA_DIR / MASKS_FILENAME,
        transform=val_transforms,
        indices=val_indices
    )

    # Create DataLoaders (Removed the duplicate definition)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize the Model, Loss, and Optimizer
    print("Initializing U-Net Architecture...")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. The Core Training Loop
    print(f"Beginning training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()

                # Calculate Dice Score for validation
                probs = torch.sigmoid(predictions)
                probs_flat = probs.view(-1)
                masks_flat = masks.view(-1)

                intersection = (probs_flat * masks_flat).sum()
                dice = (2.0 * intersection + 1e-6) / (probs_flat.sum() + masks_flat.sum() + 1e-6)
                val_dice += dice.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f}")

    # 4. Save the Model Weights
    print("--- Training Complete ---")
    # FIXED: Use PROCESSED_DIR from config.py instead of hardcoded "data/processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_DIR / WEIGHTS_FILENAME
    torch.save(model.state_dict(), save_path)
    print(f"Model weights successfully saved to: {save_path}")


if __name__ == "__main__":
    train_model()