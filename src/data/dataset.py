import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Callable


class LBNLTEMDataset(Dataset):
    """
    PyTorch Dataset for loading LBNL High-Resolution TEM images and segmentation masks
    from separate HDF5 files.
    """

    def __init__(self, images_path: str | Path, masks_path: str | Path, transform: Optional[Callable] = None) -> None:
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.transform = transform

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images dataset not found at {self.images_path}")
        if not self.masks_path.exists():
            raise FileNotFoundError(f"Masks dataset not found at {self.masks_path}")

        # Briefly open to extract the total number of samples and determine dynamic HDF5 keys
        with h5py.File(self.images_path, 'r') as f_img:
            self.img_key = list(f_img.keys())[0]  # Dynamically grab the root key
            self.length = len(f_img[self.img_key])

        with h5py.File(self.masks_path, 'r') as f_mask:
            self.mask_key = list(f_mask.keys())[0]
            assert len(
                f_mask[self.mask_key]) == self.length, "Images and masks must have the exact same number of samples."

        # Initialize as None to guarantee safe multiprocessing in DataLoaders
        self.h5_images: Optional[h5py.File] = None
        self.h5_masks: Optional[h5py.File] = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy loading: Files are only opened when a worker explicitly requests a batch
        if self.h5_images is None:
            self.h5_images = h5py.File(self.images_path, 'r')
        if self.h5_masks is None:
            self.h5_masks = h5py.File(self.masks_path, 'r')

        image = self.h5_images[self.img_key][idx]
        mask = self.h5_masks[self.mask_key][idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # PyTorch expects Channels-First tensors (CxHxW)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            else:
                image = image.transpose(2, 0, 1)

            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)

            # Normalize images to [0, 1], masks stay as 0/1 categorical integers
            image = torch.from_numpy(image).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return image, mask