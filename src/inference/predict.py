from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from src.data.transforms import get_inference_transforms
from src.models.unet import UNet
from src.config import RAW_DATA_DIR, PROCESSED_DIR, IMAGES_FILENAME, MASKS_FILENAME, WEIGHTS_FILENAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = RAW_DATA_DIR / IMAGES_FILENAME
MASK_PATH = RAW_DATA_DIR / MASKS_FILENAME
WEIGHTS_PATH = PROCESSED_DIR / WEIGHTS_FILENAME
OUTPUT_DIR = PROCESSED_DIR / "inference_debug"

IMAGE_SIZE = 256
THRESHOLD = 0.80

# Testing the 26 held-out validation images (indices 102 to 127)
IMAGE_INDICES = list(range(102, 128))

# Cache for HDF5 keys to prevent scanning the file 52 times
_H5_KEY_CACHE = {}


def get_dataset_key(file_path: str) -> str:
    """Finds the first dataset key in an HDF5 file and caches it."""
    if file_path not in _H5_KEY_CACHE:
        with h5py.File(file_path, "r") as f:
            keys = []

            # Only scan until we find the first dataset
            def find_first(name, obj):
                if isinstance(obj, h5py.Dataset) and not keys:
                    keys.append(name)

            f.visititems(find_first)
            if not keys:
                raise RuntimeError(f"No datasets found in {file_path}")
            _H5_KEY_CACHE[file_path] = keys[0]
            print(f"Indexed HDF5 key for {Path(file_path).name}: {keys[0]}")
    return _H5_KEY_CACHE[file_path]


def load_h5_array(file_path: str, index: int) -> np.ndarray:
    key = get_dataset_key(file_path)
    with h5py.File(file_path, "r") as f:
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
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(model: UNet, image_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    print(
        f"  -> Prob min: {probabilities.min().item():.4f} | max: {probabilities.max().item():.4f} | mean: {probabilities.mean().item():.4f}")

    probability_map = probabilities.squeeze().cpu().numpy()
    binary_mask = (probability_map > THRESHOLD).astype(np.uint8)
    return probability_map, binary_mask


def save_debug_figure_cv2(
        index: int,
        image: np.ndarray,
        ground_truth: np.ndarray,
        probability_map: np.ndarray,
        binary_mask: np.ndarray,
) -> None:
    """Saves a 4-panel debug image using OpenCV (bypasses Matplotlib freezes)."""

    # Helper to convert any array to 8-bit grayscale for OpenCV
    def to_uint8(img):
        if img.max() <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    img_disp = to_uint8(image)
    gt_disp = to_uint8(ground_truth)
    prob_disp = to_uint8(probability_map)
    mask_disp = binary_mask * 255

    # Add simple text labels using OpenCV
    def add_label(img, text):
        return cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    img_disp = add_label(img_disp, "Input")
    gt_disp = add_label(gt_disp, "Ground Truth")
    prob_disp = add_label(prob_disp, "Probability")
    mask_disp = add_label(mask_disp, f"Pred > {THRESHOLD}")

    # Stack the 4 images horizontally
    combined = np.hstack([img_disp, gt_disp, prob_disp, mask_disp])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"inference_debug_index_{index}.png"
    cv2.imwrite(str(output_path), combined)
    print(f"  -> Saved: {output_path}")


def main() -> None:
    print(f"Running TEM inference debug on {DEVICE}")

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    model = load_model(WEIGHTS_PATH)
    inference_transform = get_inference_transforms(crop_size=IMAGE_SIZE)

    for index in IMAGE_INDICES:
        print(f"\n[Image {index}/127]")

        image = load_h5_array(str(IMAGE_PATH), index)
        mask = load_h5_array(str(MASK_PATH), index)
        mask = normalize_mask(mask)

        transformed = inference_transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        image_display = image_tensor.squeeze().cpu().numpy()
        mask_display = mask_tensor.squeeze().cpu().numpy()

        gt_mean = mask_display.mean()
        print(f"  -> GT Mean (Nanoparticle coverage): {gt_mean:.4f}")

        probability_map, binary_mask = run_inference(model, image_tensor)

        save_debug_figure_cv2(
            index=index,
            image=image_display,
            ground_truth=mask_display,
            probability_map=probability_map,
            binary_mask=binary_mask,
        )


if __name__ == "__main__":
    main()