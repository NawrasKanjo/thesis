import os
import pandas as pd
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("results_corrected/model_performance_results.csv")
df = df.sort_values(by=["Model", "Noise_Level"])

# === Output Folder ===
output_dir = "results_graphs"
os.makedirs(output_dir, exist_ok=True)

# === Plotting Function ===
def line_plot(metric, ylabel, filename):
    if metric not in df.columns:
        print(f"⚠️ Skipping '{metric}' — not found in data.")
        return

    plt.figure(figsize=(10, 6))
    for model in df["Model"].unique():
        sub = df[df["Model"] == model]
        plt.plot(sub["Noise_Level"], sub[metric], marker='o', label=model)

    plt.title(f"{ylabel} vs Noise Level", fontsize=14)
    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Model", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# === Metrics to Plot ===

# H1: Model Performance
line_plot("Accuracy", "Accuracy", "accuracy_vs_noise.png")
line_plot("F1_Score", "F1 Score", "f1_score_vs_noise.png")
line_plot("Recall", "Recall", "recall_vs_noise.png")
line_plot("Precision", "Precision", "precision_vs_noise.png")

# H2: Sustainability Metrics
line_plot("CO2e_kg", "CO₂ Emissions (kg, CodeCarbon)", "co2_vs_noise.png")
line_plot("Energy_kWh", "Energy (kWh, Intel Power Gadget)", "ipg_energy_vs_noise.png")
line_plot("Train_Time_sec", "Training Time (seconds)", "train_time_vs_noise.png")
line_plot("Inference_Time_ms", "Inference Time (ms)", "inference_time_vs_noise.png")
line_plot("RAM_MB", "Peak RAM Usage (MB)", "ram_usage_vs_noise.png")
line_plot("Model_Size_MB", "Model Size (MB)", "model_size_vs_noise.png")

# H3: Tradeoff Metrics
line_plot("Accuracy_per_CO2", "Accuracy per kg CO₂", "accuracy_per_co2_vs_noise.png")
line_plot("F1_Score_per_CO2", "F1 Score per kg CO₂", "f1_per_co2_vs_noise.png")
line_plot("Accuracy_per_Time", "Accuracy per second", "accuracy_per_time_vs_noise.png")
line_plot("Accuracy_per_RAM", "Accuracy per MB RAM", "accuracy_per_ram_vs_noise.png")

# === Save processed copy for reference ===
df.to_csv(os.path.join(output_dir, "processed_results.csv"), index=False)
print("✓ Graphs generated and saved to:", output_dir)
