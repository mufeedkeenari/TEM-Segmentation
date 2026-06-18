import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The fundamental building block of a U-Net.
    It applies two Convolutional layers, each followed by Batch Normalization and a ReLU activation.
    Convolutions extract the physical features (edges, gradients) of the nanoparticles.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second Convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    The complete U-Net architecture for TEM Semantic Segmentation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        # in_channels = 1 because TEM images are grayscale.
        # out_channels = 1 because we want a single binary mask (Nanoparticle vs. Background).
        super().__init__()

        # --- THE ENCODER (Going Down) ---
        # Extracts features while shrinking the image resolution.
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- THE DECODER (Going Up) ---
        # ConvTranspose2d doubles the image size.
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # --- THE OUTPUT LAYER ---
        # Maps the 64 feature channels down to 1 single prediction channel.
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Forward Pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder Forward Pass with SKIP CONNECTIONS
        # torch.cat() bridges the high-resolution features from the encoder directly to the decoder.
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)  # Skip Connection 1
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)  # Skip Connection 2
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)  # Skip Connection 3
        x = self.conv3(x)

        # Final prediction (Raw logits, not yet bounded between 0 and 1)
        logits = self.outc(x)
        return logits