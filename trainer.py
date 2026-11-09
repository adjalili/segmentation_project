import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from args import DEVICE, EPOCHS, LEARNING_RATE
from utils import dice_score, dice_loss


def train_model(model, train_loader, val_loader):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    bce_loss = nn.BCEWithLogitsLoss()

    best_dice = 0.0

    # Create checkpoint and results folders
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Store training history
    train_history = []
    val_history = []
    dice_history = []

    for epoch in range(EPOCHS):

        # Training
        model.train()

        train_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            predictions = model(images)

            loss = (
                bce_loss(predictions, masks)
                + dice_loss(predictions, masks)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()

        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                predictions = model(images)

                loss = (
                    bce_loss(predictions, masks)
                    + dice_loss(predictions, masks)
                )

                val_loss += loss.item()
                val_dice += dice_score(predictions, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Save training history
        train_history.append(train_loss)
        val_history.append(val_loss)
        dice_history.append(val_dice)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save the best model
        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

            print("Best model saved!")

    # Save training history
    history = {
        "train_loss": train_history,
        "val_loss": val_history,
        "val_dice": dice_history
    }

    torch.save(
        history,
        "checkpoints/training_history.pth"
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        train_history,
        label="Training Loss"
    )

    plt.plot(
        range(1, EPOCHS + 1),
        val_history,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/training_loss.png", dpi=150)
    plt.close()

    # Plot validation Dice score
    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        dice_history,
        label="Validation Dice"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.savefig("results/validation_dice.png", dpi=150)
    plt.close()

    print("\nTraining finished!")
    print("Best validation Dice:", round(best_dice, 4))
    print("Training graphs saved in results/")

    # Load the best model
    if os.path.exists("checkpoints/best_model.pth"):

        model.load_state_dict(
            torch.load(
                "checkpoints/best_model.pth",
                map_location=DEVICE,
                weights_only=True
            )
        )

    return model
