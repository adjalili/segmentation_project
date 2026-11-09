
import torch

from args import VAL_CSV, DEVICE
from dataset import KneeDataset
from model import UNet
from evaluate import visualize_validation


# Saved model checkpoint
CHECKPOINT_PATH = "checkpoints/best_model.pth"


def main():

    # Load validation dataset
    val_dataset = KneeDataset(VAL_CSV)

    print("Validation samples:", len(val_dataset))

    # Initialize U-Net
    model = UNet().to(DEVICE)

    # Load previously trained model
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    print("Saved model loaded successfully!")

    # Generate validation visualizations
    visualize_validation(
        model,
        val_dataset,
        num_images=5
    )

    print("Validation images regenerated without training!")


if __name__ == "__main__":
    main()
