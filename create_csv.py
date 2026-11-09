
from pathlib import Path
import pandas as pd
import re

# 1. Define dataset directories
XRAY_DIR = Path("dataset/xrays")
MASK_DIR = Path("dataset/masks")
CSV_DIR = Path("dataset/CSVs")

# 2. Find X-ray images
xray_files = sorted(
    XRAY_DIR.glob("*.png"),
    key=lambda p: int(p.stem)
)

# 3. Build a dictionary of masks using image numbers
mask_lookup = {}

for mask_path in MASK_DIR.glob("*.png"):
    match = re.match(r"^(\d+)(?:\.png|_png\.rf\..*_mask\.png)$", mask_path.name)

    if match:
        image_number = int(match.group(1))
        mask_lookup[image_number] = mask_path

# 4. Build the main dataset
rows = []

for xray_path in xray_files:
    image_number = int(xray_path.stem)

    mask_path = mask_lookup.get(image_number)

    rows.append({
        "xrays": str(xray_path),
        "masks": str(mask_path) if mask_path else "None"
    })

dataset_df = pd.DataFrame(rows)

# 5. Separate labeled and test images
labeled_df = dataset_df[dataset_df["masks"] != "None"].copy()
test_df = dataset_df[dataset_df["masks"] == "None"].copy()

# 6. Randomly split labeled images (80% / 20%)
train_df = labeled_df.sample(frac=0.8, random_state=42)
validation_df = labeled_df.drop(train_df.index)

# 7. Create output directory
CSV_DIR.mkdir(parents=True, exist_ok=True)

# 8. Save CSV files
dataset_df.to_csv(CSV_DIR / "dataset.csv", index=False)
train_df.to_csv(CSV_DIR / "train.csv", index=False)
validation_df.to_csv(CSV_DIR / "validation.csv", index=False)
test_df.to_csv(CSV_DIR / "test.csv", index=False)

# 9. Display results
print("Dataset CSV files created successfully!")

print("Total images:", len(dataset_df))
print("Training images:", len(train_df))
print("Validation images:", len(validation_df))
print("Test images:", len(test_df))
