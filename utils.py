
import torch


def dice_score(predictions, targets, smooth=1e-6):

    # Convert logits to binary masks
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()

    # Calculate Dice for each image
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


def dice_loss(predictions, targets, smooth=1e-6):

    # Convert logits to probabilities
    predictions = torch.sigmoid(predictions)

    # Calculate Dice for each image
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice.mean()
