import pandas as pd
import numpy as np
import os

INPUT_FILE = "../../../../Datasets/Selected rows/walk_run/dataset.csv"
OUTPUT_DIR = "../../../../Datasets/Selected rows/walk_run/noisy_datasets"

COLUMNS = [
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]

NOISE_LEVELS = list(range(0, 101, 10))  # 0%, 5%, ..., 100%

def inject_noise(df, columns, noise_percent):
    noisy_df = df.copy()
    n_rows = len(df)
    n_noisy = int((noise_percent / 100.0) * n_rows)

    if noise_percent == 0:
        return noisy_df

    # Select same rows across all columns
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)

    for col in columns:
        std = df[col].std()
        noise = np.random.normal(loc=0.0, scale=std, size=n_noisy)
        noisy_df.loc[noisy_indices, col] += noise

        # Clip values to original range
        col_min = df[col].min()
        col_max = df[col].max()
        noisy_df[col] = np.clip(noisy_df[col], col_min, col_max)

    return noisy_df


def main():
    df = pd.read_csv(INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in NOISE_LEVELS:
        noisy_df = inject_noise(df, COLUMNS, level)
        output_path = os.path.join(OUTPUT_DIR, f"dataset_{level}.csv")
        noisy_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    main()
