from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "data/raw/au_10nm_images.h5"
WEIGHTS_PATH = "data/processed/unet_tem_weights.pth"
OUTPUT_PATH = "data/processed/inference_preview.png"

IMAGE_INDEX = 0
IMAGE_SIZE = 256
THRESHOLD = 0.40


def find_first_dataset_key(h5_file: h5py.File) -> str:
    dataset_keys = []

    def collect_dataset_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_keys.append(name)

    h5_file.visititems(collect_dataset_keys)

    if not dataset_keys:
        raise RuntimeError("No datasets found inside the HDF5 file.")

    print(f"Available HDF5 datasets: {dataset_keys}")
    print(f"Using dataset: {dataset_keys[0]}")

    return dataset_keys[0]


def load_h5_image(image_path: str, image_index: int) -> tuple[torch.Tensor, np.ndarray]:
    with h5py.File(image_path, "r") as f:
        key = find_first_dataset_key(f)
        image = f[key][image_index]

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")

    if image.max() > 1.0:
        image = image / 255.0

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    resized_image = image_tensor.squeeze().numpy()

    return image_tensor, resized_image


def load_model(weights_path: str) -> UNet:
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def run_inference(model: UNet, image_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        print(f"Probability min:  {probabilities.min().item():.4f}")
        print(f"Probability max:  {probabilities.max().item():.4f}")
        print(f"Probability mean: {probabilities.mean().item():.4f}")
        print(f"Foreground pixels at threshold {THRESHOLD}: {(probabilities > THRESHOLD).float().mean().item():.6f}")
        binary_mask = probabilities > THRESHOLD

    probability_map = probabilities.squeeze().cpu().numpy()
    binary_mask = binary_mask.squeeze().cpu().numpy().astype(np.uint8)

    return probability_map, binary_mask


def save_inference_preview(
    image: np.ndarray,
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input TEM Image")
    axes[0].axis("off")

    axes[1].imshow(probability_map, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted Probability")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Binary Mask > {THRESHOLD}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Inference preview saved to: {output_path}")


def main() -> None:
    print(f"Running TEM inference on {DEVICE}")

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"HDF5 image file not found: {IMAGE_PATH}")

    model = load_model(WEIGHTS_PATH)
    image_tensor, image = load_h5_image(IMAGE_PATH, IMAGE_INDEX)

    probability_map, binary_mask = run_inference(model, image_tensor)

    save_inference_preview(
        image=image,
        probability_map=probability_map,
        binary_mask=binary_mask,
        output_path=OUTPUT_PATH,
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()