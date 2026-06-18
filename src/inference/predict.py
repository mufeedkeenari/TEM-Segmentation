from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import get_inference_transforms
from src.models.unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "data/raw/au_10nm_images.h5"
MASK_PATH = "data/raw/au_10nm_labels.h5"
WEIGHTS_PATH = "data/processed/unet_tem_weights.pth"

OUTPUT_DIR = "data/processed/inference_debug"

IMAGE_SIZE = 256
THRESHOLD = 0.8

IMAGE_INDICES = [0, 1, 5, 10, 20]


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


def load_h5_array(file_path: str, index: int) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        key = find_first_dataset_key(f)
        array = f[key][index]

    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {array.shape}")

    return array


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)

    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)

    return mask


def load_model(weights_path: str) -> UNet:
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def run_inference(model: UNet, image_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    print(f"Probability min:  {probabilities.min().item():.4f}")
    print(f"Probability max:  {probabilities.max().item():.4f}")
    print(f"Probability mean: {probabilities.mean().item():.4f}")
    print(
        f"Foreground fraction at threshold {THRESHOLD}: "
        f"{(probabilities > THRESHOLD).float().mean().item():.6f}"
    )

    probability_map = probabilities.squeeze().cpu().numpy()
    binary_mask = (probability_map > THRESHOLD).astype(np.uint8)

    return probability_map, binary_mask


def save_debug_figure(
    index: int,
    image: np.ndarray,
    ground_truth: np.ndarray,
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    output_path = Path(OUTPUT_DIR) / f"inference_debug_index_{index}.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Input Image #{index}")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(probability_map, cmap="gray")
    axes[2].set_title("Probability Map")
    axes[2].axis("off")

    axes[3].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction > {THRESHOLD}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved debug figure to: {output_path}")
def compute_metrics(
    probability_map: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: list[float],
) -> None:
    gt = ground_truth.astype(bool)

    print("Threshold sweep:")
    print("threshold | pred_frac | dice   | iou    | false_pos_frac")

    for threshold in thresholds:
        pred = probability_map > threshold

        intersection = np.logical_and(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        union = np.logical_or(pred, gt).sum()

        dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
        iou = intersection / (union + 1e-8)

        false_positive = np.logical_and(pred, ~gt).sum()
        false_positive_fraction = false_positive / pred.size

        print(
            f"{threshold:8.2f} | "
            f"{pred.mean():9.6f} | "
            f"{dice:6.4f} | "
            f"{iou:6.4f} | "
            f"{false_positive_fraction:14.6f}"
        )

def main() -> None:
    print(f"Running TEM inference debug on {DEVICE}")

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image HDF5 file not found: {IMAGE_PATH}")

    if not Path(MASK_PATH).exists():
        raise FileNotFoundError(f"Mask HDF5 file not found: {MASK_PATH}")

    model = load_model(WEIGHTS_PATH)
    inference_transform = get_inference_transforms(crop_size=IMAGE_SIZE)

    for index in IMAGE_INDICES:
        print("\n" + "=" * 80)
        print(f"Running inference for image index: {index}")

        image = load_h5_array(IMAGE_PATH, index)
        mask = load_h5_array(MASK_PATH, index)
        mask = normalize_mask(mask)

        transformed = inference_transform(image=image, mask=mask)

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        image_display = image_tensor.squeeze().cpu().numpy()
        mask_display = mask_tensor.squeeze().cpu().numpy()

        print(
            f"Image tensor shape: {image_tensor.shape}, "
            f"min={image_tensor.min().item():.4f}, "
            f"max={image_tensor.max().item():.4f}, "
            f"mean={image_tensor.mean().item():.4f}, "
            f"std={image_tensor.std().item():.4f}"
        )

        print(
            f"Mask tensor shape: {mask_tensor.shape}, "
            f"min={mask_tensor.min().item():.4f}, "
            f"max={mask_tensor.max().item():.4f}, "
            f"mean={mask_tensor.float().mean().item():.6f}"
        )

        probability_map, binary_mask = run_inference(model, image_tensor)
        compute_metrics(
            probability_map=probability_map,
            ground_truth=mask_display,
            thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        )
        save_debug_figure(
            index=index,
            image=image_display,
            ground_truth=mask_display,
            probability_map=probability_map,
            binary_mask=binary_mask,
        )


if __name__ == "__main__":
    main()