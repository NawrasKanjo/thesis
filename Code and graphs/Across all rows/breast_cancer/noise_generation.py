import pandas as pd
import numpy as np
import os

INPUT_FILE = "dataset.data"
OUTPUT_DIR = "noisy_datasets"

# Columns from index 2 to 31 (exclude ID and binary target 'Diagnosis')
COLUMNS = list(range(2, 32))

NOISE_LEVELS = list(range(0, 101, 10))  # 0%, 10%, ..., 100%

def inject_noise(df, columns, noise_percent):
    noisy_df = df.copy()

    for col in columns:
        std = df[col].std()
        noise = np.random.normal(loc=0.0, scale=std * (noise_percent / 100.0), size=len(df))
        noisy_df[col] += noise

        col_min = df[col].min()
        col_max = df[col].max()
        noisy_df[col] = np.clip(noisy_df[col], col_min, col_max)

    return noisy_df



def main():
    # Load dataset without headers
    df = pd.read_csv(INPUT_FILE, header=None)

    # Convert target (column 1) to binary if not already done
    if df[1].dtype == object:
        df[1] = df[1].map({'B': 0, 'M': 1})

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in NOISE_LEVELS:
        noisy_df = inject_noise(df, COLUMNS, level)
        output_path = os.path.join(OUTPUT_DIR, f"dataset_{level}.csv")
        noisy_df.to_csv(output_path, index=False, header=False)
        print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    main()
