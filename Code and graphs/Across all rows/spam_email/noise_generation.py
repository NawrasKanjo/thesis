import pandas as pd
import numpy as np
import os

INPUT_FILE = "dataset.csv"
OUTPUT_DIR = "noisy_datasets"

COLUMNS = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
]

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
    df = pd.read_csv(INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in NOISE_LEVELS:
        noisy_df = inject_noise(df, COLUMNS, level)
        output_path = os.path.join(OUTPUT_DIR, f"dataset_{level}.csv")
        noisy_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    main()
