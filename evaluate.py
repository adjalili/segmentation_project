
import os
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from args import DEVICE
from utils import dice_score


def evaluate_model(model, val_loader):

    model.eval()

    total_dice = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            predictions = model(images)

            total_dice += dice_score(predictions, masks)

    average_dice = total_dice / len(val_loader)

    print("Validation Dice:", round(average_dice, 4))

    return average_dice


def visualize_validation(model, val_dataset, num_images=5):

    model.eval()

    os.makedirs("results/validation", exist_ok=True)

    for index in range(min(num_images, len(val_dataset))):

        sample = val_dataset[index]

        image = sample["image"].unsqueeze(0).to(DEVICE)
        target = sample["mask"].squeeze().numpy()

        # Predict femur mask
        with torch.no_grad():

            prediction = model(image)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()

        prediction = prediction.squeeze().cpu().numpy()

        # Prepare original X-ray
        xray = (
            sample["image"].squeeze().numpy() * 255
        ).astype("uint8")

        # Create overlay
        overlay = cv2.cvtColor(xray, cv2.COLOR_GRAY2RGB)

        overlay[prediction > 0] = (
                overlay[prediction > 0].astype("float32") * 0.4
                + np.array([255, 0, 0]) * 0.6
        ).astype("uint8")

        # Display comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(xray, cmap="gray")
        axes[0].set_title("Original X-ray")

        axes[1].imshow(target, cmap="gray")
        axes[1].set_title("Ground truth")

        axes[2].imshow(prediction, cmap="gray")
        axes[2].set_title("Predicted femur")

        axes[3].imshow(overlay)
        axes[3].set_title("Prediction overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"results/validation/sample_{index}.png"
        )

        plt.close()

    print("Validation visualizations saved!")


def predict_test_images(model, test_csv):

    model.eval()

    os.makedirs("results", exist_ok=True)

    data = pd.read_csv(test_csv)

    with torch.no_grad():

        for index, row in data.iterrows():

            xray_path = row["xrays"]

            # Load X-ray
            xray = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)

            if xray is None:
                raise FileNotFoundError(
                    f"Could not load X-ray: {xray_path}"
                )

            # Preprocess image
            image = xray.astype("float32") / 255.0
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image = image.to(DEVICE)

            # Predict femur mask
            prediction = model(image)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()

            mask = prediction.squeeze().cpu().numpy()
            mask = (mask * 255).astype("uint8")

            # Create overlay
            overlay = cv2.cvtColor(xray, cv2.COLOR_GRAY2BGR)
            overlay[mask > 0] = (0, 0, 255)

            result = cv2.addWeighted(
                cv2.cvtColor(xray, cv2.COLOR_GRAY2BGR),
                0.7,
                overlay,
                0.3,
                0
            )

            # Save results
            filename = os.path.basename(xray_path)

            cv2.imwrite(
                f"results/{filename}",
                result
            )

            cv2.imwrite(
                f"results/{os.path.splitext(filename)[0]}_mask.png",
                mask
            )

    print("Test predictions saved in results/")
