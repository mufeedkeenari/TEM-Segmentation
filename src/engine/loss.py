import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Expects:
        logits: [B, 1, H, W]
        true_masks: [B, 1, H, W]
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        true_flat = true_masks.view(-1)

        intersection = (probs_flat * true_flat).sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + true_flat.sum() + self.smooth
        )

        return 1.0 - dice_coeff


class BCEDiceLoss(nn.Module):
    """
    Hybrid BCE + Dice loss for binary segmentation.

    BCEWithLogitsLoss provides stable pixel-wise gradients.
    Dice loss directly optimizes mask overlap.

    Expects:
        logits: [B, 1, H, W]
        true_masks: [B, 1, H, W]
    """

    def __init__(self, smooth: float = 1e-6, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, true_masks)

        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        true_flat = true_masks.view(-1)

        intersection = (probs_flat * true_flat).sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + true_flat.sum() + self.smooth
        )

        dice_loss = 1.0 - dice_coeff

        return self.bce_weight * bce_loss + dice_loss