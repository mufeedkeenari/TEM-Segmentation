from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LBNLTEMDataset(Dataset):
    """
    PyTorch Dataset for loading LBNL HRTEM images and segmentation masks
    from separate HDF5 files.

    Returns
    -------
    image : torch.Tensor
        Grayscale image tensor with shape [1, H, W].

    mask : torch.Tensor
        Binary segmentation mask tensor with shape [1, H, W].
    """

    def __init__(
        self,
        images_path: str | Path,
        masks_path: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.transform = transform

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images dataset not found at {self.images_path}")

        if not self.masks_path.exists():
            raise FileNotFoundError(f"Masks dataset not found at {self.masks_path}")

        # Open briefly to read dataset keys and sample count.
        # The files themselves are not kept open here.
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
        # These are opened only inside __getitem__, which is safer with DataLoader workers.
        self.h5_images: Optional[h5py.File] = None
        self.h5_masks: Optional[h5py.File] = None

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _prepare_image_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Convert image to a float tensor with shape [1, H, W] or [C, H, W].

        If Albumentations ToTensorV2 has already run, image should already be
        a torch.Tensor in channel-first format.
        """
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            elif image.ndim == 3:
                # Convert HWC to CHW if needed.
                if image.shape[-1] in (1, 3):
                    image = image.transpose(2, 0, 1)
            else:
                raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

            image = torch.from_numpy(image).float()

            # Default normalization for non-transform path.
            # If values are already normalized or standardized, do not force rescaling.
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
        """
        Convert mask to a binary float tensor with shape [1, H, W].
        """
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)

            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            elif mask.ndim == 3:
                # Convert HWC mask to CHW if needed.
                if mask.shape[-1] == 1:
                    mask = mask.transpose(2, 0, 1)
                elif mask.shape[0] != 1:
                    raise ValueError(
                        "Expected binary mask with one channel. "
                        f"Got mask shape {mask.shape}"
                    )
            else:
                raise ValueError(f"Expected 2D or 3D mask array, got shape {mask.shape}")

            mask = torch.from_numpy(mask).float()

        elif isinstance(mask, torch.Tensor):
            mask = mask.float()

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                if mask.shape[0] != 1:
                    raise ValueError(
                        "Expected binary mask with one channel. "
                        f"Got mask tensor shape {mask.shape}"
                    )
            else:
                raise ValueError(f"Expected 2D or 3D mask tensor, got shape {mask.shape}")

        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        # Convert 0/255 masks to 0/1.
        if mask.max() > 1.0:
            mask = mask / 255.0

        # Force binary target.
        mask = (mask > 0.5).float()

        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy loading: files are opened only when a sample is requested.
        if self.h5_images is None:
            self.h5_images = h5py.File(self.images_path, "r")

        if self.h5_masks is None:
            self.h5_masks = h5py.File(self.masks_path, "r")

        image = self.h5_images[self.img_key][idx]
        mask = self.h5_masks[self.mask_key][idx]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = self._prepare_image_tensor(image)
        mask = self._prepare_mask_tensor(mask)

        return image, mask