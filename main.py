
from torch.utils.data import DataLoader

from args import BATCH_SIZE, TRAIN_CSV, VAL_CSV, TEST_CSV
from dataset import KneeDataset
from model import UNet
from trainer import train_model
from evaluate import (
    evaluate_model,
    visualize_validation,
    predict_test_images
)


def main():

    # Load datasets
    train_dataset = KneeDataset(TRAIN_CSV)
    val_dataset = KneeDataset(VAL_CSV)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Initialize U-Net
    model = UNet()

    # Train model
    model = train_model(
        model,
        train_loader,
        val_loader
    )

    # Evaluate on validation data
    evaluate_model(model, val_loader)

    # Visualize validation predictions
    visualize_validation(
        model,
        val_dataset,
        num_images=5
    )

    # Predict separate test images
    predict_test_images(model, TEST_CSV)


if __name__ == "__main__":
    main()
