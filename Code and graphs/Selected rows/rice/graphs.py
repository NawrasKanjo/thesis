import os
import pandas as pd
import matplotlib.pyplot as plt

# Load and sort data
df = pd.read_csv("results_corrected/model_performance_results_corrected.csv")
df = df.sort_values(by=["Model", "Noise_Level"])

# Output directory
output_dir = "graphs_thesis_ready"
os.makedirs(output_dir, exist_ok=True)

# --- Line Plot Function ---
def line_plot(metric, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for model in df["Model"].unique():
        sub_df = df[df["Model"] == model]
        plt.plot(
            sub_df["Noise_Level"],
            sub_df[metric],
            label=model,
            marker='o',
            linestyle='-',
            linewidth=2
        )
    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs Noise Level (Rice)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# --- PERFORMANCE METRICS (H1) ---
line_plot("Accuracy", "Accuracy", "accuracy_vs_noise.png")
line_plot("F1_Score", "F1 Score", "f1_score_vs_noise.png")
line_plot("Recall", "Recall", "recall_vs_noise.png")
line_plot("Precision", "Precision", "precision_vs_noise.png")

# --- SUSTAINABILITY METRICS (H2) ---
line_plot("CO2e_kg", "CO₂ Emissions (kg)", "co2_vs_noise.png")
#line_plot("Energy_kWh", "Energy (kWh)", "energy_vs_noise.png")
line_plot("Train_Time_sec", "Training Time (s)", "train_time_vs_noise.png")
line_plot("RAM_MB", "RAM Usage (MB)", "ram_vs_noise.png")

# --- EFFICIENCY METRICS (H3) ---
line_plot("Accuracy_per_CO2", "Accuracy per kg CO₂", "accuracy_per_co2_vs_noise.png")
line_plot("F1_Score_per_CO2", "F1 Score per kg CO₂", "f1_per_co2_vs_noise.png")

# Save sorted results
df.to_csv(os.path.join(output_dir, "merged_results_sorted.csv"), index=False)
