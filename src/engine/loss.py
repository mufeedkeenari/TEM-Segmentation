import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for Semantic Segmentation.
    Optimizes directly for the overlap between the predicted mask and the ground truth.
    Highly resistant to class imbalance (e.g., when most of a TEM image is empty background).
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): A small constant added to the numerator and denominator
                            to prevent division by zero if both masks are completely empty.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        # The U-Net outputs raw logits. We apply a Sigmoid activation to squash them
        # into probabilities between 0.0 (Background) and 1.0 (Nanoparticle).
        probs = torch.sigmoid(logits)

        # Flatten the tensors from (Batch, Channels, Height, Width) into 1D arrays
        probs_flat = probs.view(-1)
        true_flat = true_masks.view(-1)

        # Calculate the mathematical intersection (overlap)
        intersection = (probs_flat * true_flat).sum()

        # Calculate the Dice Coefficient (Formula: 2 * Overlap / (Total Pixels in A + Total Pixels in B))
        dice_coeff = (2. * intersection + self.smooth) / (probs_flat.sum() + true_flat.sum() + self.smooth)

        # The optimizer wants to MINIMIZE loss, so we return 1 minus the coefficient.
        # A perfect overlap gives a coefficient of 1.0, resulting in a loss of 0.0.
        return 1. - dice_coeff