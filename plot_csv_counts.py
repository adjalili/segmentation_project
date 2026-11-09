
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Location of the CSV files
CSV_DIR = Path("dataset/CSVs")

# Read the number of rows in each CSV
files = {
    "Dataset": "dataset.csv",
    "Training": "train.csv",
    "Validation": "validation.csv",
    "Test": "test.csv"
}

sample_counts = {}

for label, filename in files.items():
    df = pd.read_csv(CSV_DIR / filename)
    sample_counts[label] = len(df)

# Create the bar chart
plt.figure(figsize=(9, 6))

bars = plt.bar(
    sample_counts.keys(),
    sample_counts.values()
)

# Add sample counts above each bar
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.title("Knee X-ray Dataset Distribution")
plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")

plt.ylim(0, max(sample_counts.values()) + 20)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save the chart
output_path = Path("dataset/dataset_split_bar_chart.png")
plt.savefig(output_path, dpi=300)

plt.show()

print("Chart saved to:", output_path)
print("Sample counts:", sample_counts)
