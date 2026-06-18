import torch
import pytest
from src.models.unet import UNet


def test_unet_output_shape() -> None:
    """
    Tests if the U-Net architecture preserves spatial dimensions
    and outputs the correct number of channels.
    """
    # Initialize the model: 1 input channel (grayscale), 1 output channel (binary mask)
    model = UNet(in_channels=1, out_channels=1)

    # Create a dummy tensor representing a batch of 4 TEM images (256x256 pixels)
    # Shape format: (Batch, Channels, Height, Width)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 256, 256)

    # Perform a forward pass through the untrained network
    output = model(dummy_input)

    # The output MUST have the exact same geometric shape as the input
    # Expected shape: (4, 1, 256, 256)
    assert output.shape == dummy_input.shape, f"Shape mismatch: Expected {dummy_input.shape}, got {output.shape}"


def test_unet_is_trainable() -> None:
    """
    Verifies that the model has trainable parameters.
    """
    model = UNet(in_channels=1, out_channels=1)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # A standard U-Net should have millions of parameters
    assert total_params > 1_000_000, "Model has too few trainable parameters to be a U-Net."