import pytest
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from src.data.dataset import LBNLTEMDataset


@pytest.fixture
def mock_dual_h5(tmp_path: Path) -> Tuple[Path, Path]:
    """Creates temporary dual HDF5 files with dummy TEM data for isolated testing."""
    img_path = tmp_path / "mock_images.h5"
    mask_path = tmp_path / "mock_masks.h5"

    with h5py.File(img_path, 'w') as f:
        # 10 dummy grayscale images (256x256)
        f.create_dataset('sensor_data', data=np.random.randint(0, 256, size=(10, 256, 256), dtype=np.uint8))

    with h5py.File(mask_path, 'w') as f:
        # 10 corresponding dummy masks
        f.create_dataset('human_labels', data=np.random.randint(0, 2, size=(10, 256, 256), dtype=np.uint8))

    return img_path, mask_path


def test_lbnl_dataset_tensors(mock_dual_h5: Tuple[Path, Path]) -> None:
    img_path, mask_path = mock_dual_h5
    dataset = LBNLTEMDataset(images_path=img_path, masks_path=mask_path)

    # Assert correct dataset length
    assert len(dataset) == 10

    # Fetch first batch
    image, mask = dataset[0]

    # Assert PyTorch Tensor formats (Channel, Height, Width)
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)

    # Assert physical data bounds
    assert torch.max(image) <= 1.0
    assert torch.min(image) >= 0.0
    # Masks should be strictly categorical (0 or 1)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})