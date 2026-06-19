from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LBNLTEMDataset(Dataset):
    """
    PyTorch Dataset for loading LBNL HRTEM images and segmentation masks
    from separate HDF5 files.
    """

    def __init__(
        self,
        images_path: str | Path,
        masks_path: str | Path,
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.transform = transform
        self.indices = indices

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images dataset not found at {self.images_path}")

        if not self.masks_path.exists():
            raise FileNotFoundError(f"Masks dataset not found at {self.masks_path}")

        # Open briefly to read dataset keys and sample count.
        with h5py.File(self.images_path, "r") as f_img:
            self.img_key = list(f_img.keys())[0]
            self.length = len(f_img[self.img_key])

        with h5py.File(self.masks_path, "r") as f_mask:
            self.mask_key = list(f_mask.keys())[0]
            mask_length = len(f_mask[self.mask_key])

        if mask_length != self.length:
            raise ValueError(
                "Images and masks must have the same number of samples. "
                f"Got {self.length} images and {mask_length} masks."
            )

        # Lazy-loaded HDF5 handles.
        self.h5_images: Optional[h5py.File] = None
        self.h5_masks: Optional[h5py.File] = None

    def __len__(self) -> int:
        # FIXED: No more f_img variable! Just return the correct length.
        if self.indices is not None:
            return len(self.indices)
        return self.length

    @staticmethod
    def _prepare_image_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            elif image.ndim == 3:
                if image.shape[-1] in (1, 3):
                    image = image.transpose(2, 0, 1)
            else:
                raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

            image = torch.from_numpy(image).float()
            if image.max() > 1.0 and image.min() >= 0.0:
                image = image / 255.0

        elif isinstance(image, torch.Tensor):
            image = image.float()
            if image.ndim == 2:
                image = image.unsqueeze(0)
            elif image.ndim != 3:
                raise ValueError(f"Expected 2D or 3D image tensor, got shape {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return image

    @staticmethod
    def _prepare_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:
                    mask = mask.transpose(2, 0, 1)
                elif mask.shape[0] != 1:
                    raise ValueError(f"Expected binary mask with one channel. Got mask shape {mask.shape}")
            else:
                raise ValueError(f"Expected 2D or 3D mask array, got shape {mask.shape}")

            mask = torch.from_numpy(mask).float()
        elif isinstance(mask, torch.Tensor):
            mask = mask.float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                if mask.shape[0] != 1:
                    raise ValueError(f"Expected binary mask with one channel. Got mask tensor shape {mask.shape}")
            else:
                raise ValueError(f"Expected 2D or 3D mask tensor, got shape {mask.shape}")
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        if mask.max() > 1.0:
            mask = mask / 255.0

        mask = (mask > 0.5).float()
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map the dataset index to the actual HDF5 index
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx

        # Lazy loading...
        if self.h5_images is None:
            self.h5_images = h5py.File(self.images_path, "r")
        if self.h5_masks is None:
            self.h5_masks = h5py.File(self.masks_path, "r")

        image = self.h5_images[self.img_key][actual_idx]
        mask = self.h5_masks[self.mask_key][actual_idx]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = self._prepare_image_tensor(image)
        mask = self._prepare_mask_tensor(mask)

        return image, mask