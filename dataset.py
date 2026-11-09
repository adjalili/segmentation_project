
import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class KneeDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        required_columns = ["xrays", "masks"]

        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(f"Missing required CSV column: {column}")

        if len(self.data) == 0:
            raise ValueError("Empty CSV file")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        xray_path = row["xrays"]
        mask_path = row["masks"]

        # Load images
        xray = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if xray is None:
            raise FileNotFoundError(f"Could not load X-ray image: {xray_path}")

        if mask is None:
            raise FileNotFoundError(f"Could not load mask image: {mask_path}")

        if xray.shape != mask.shape:
            raise ValueError(
                f"Image and mask dimensions do not match: "
                f"{xray.shape} vs {mask.shape}"
            )

        # Normalize image and convert mask to binary
        xray = xray.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension
        xray = np.expand_dims(xray, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        xray = torch.from_numpy(xray)
        mask = torch.from_numpy(mask)

        return {
            "image": xray,
            "mask": mask
        }
