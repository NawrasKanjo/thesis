import pandas as pd
import numpy as np
import os

INPUT_FILE = "../../../../Datasets/Across all rows/credit_card_fraud/dataset.csv"
OUTPUT_DIR = "noisy_datasets"

COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

NOISE_LEVELS = list(range(0, 101, 10))  # 0% to 100% noise in steps

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
    df = pd.read_csv(INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in NOISE_LEVELS:
        noisy_df = inject_noise(df, COLUMNS, level)
        output_path = os.path.join(OUTPUT_DIR, f"dataset_{level}.csv")
        noisy_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    main()
