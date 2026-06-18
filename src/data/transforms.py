import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(crop_size: int = 256) -> A.Compose:
    """
    Data augmentations for the training set.
    Injects geometric variance to prevent overfitting on small TEM datasets.
    """
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.ToFloat normalizes the numpy array by dividing by 255.0
        A.ToFloat(max_value=255.0),
        # ToTensorV2 converts to PyTorch standard (Channels, Height, Width)
        ToTensorV2()
    ])

def get_val_transforms(crop_size: int = 256) -> A.Compose:
    """
    Transforms for the validation set.
    Strictly deterministic to ensure consistent evaluation metrics.
    """
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])

def get_inference_transforms(crop_size: int = 256) -> A.Compose:
    """
    Deterministic transforms for inference.

    Uses the same intensity normalization convention as training/validation,
    but removes random augmentations.
    """
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])