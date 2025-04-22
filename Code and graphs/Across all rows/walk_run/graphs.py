import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reset all styles to default
plt.style.use('default')

# Set Seaborn style
sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.9'})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])

# === Load Data ===
df = pd.read_csv("results_corrected/model_performance_results.csv")
df = df.sort_values(by=["Model", "Noise_Level"])

# === Output Folders ===
# Create main directories
output_dir = "results_graphs"
log_output_dir = "log_graphs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_output_dir, exist_ok=True)

# Create subdirectories for each section
accuracy_dir = os.path.join(output_dir, "accuracy")
sustainability_dir = os.path.join(output_dir, "sustainability")
log_accuracy_dir = os.path.join(log_output_dir, "accuracy")
log_sustainability_dir = os.path.join(log_output_dir, "sustainability")

# Create all directories
for dir_path in [accuracy_dir, sustainability_dir, log_accuracy_dir, log_sustainability_dir]:
    os.makedirs(dir_path, exist_ok=True)

# === Plotting Functions ===
def line_plot(metric, ylabel, filename, section="accuracy"):
    if metric not in df.columns:
        print(f"⚠️ Skipping '{metric}' — not found in data.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the line plot
    ax = sns.lineplot(data=df, x="Noise_Level", y=metric, hue="Model", 
                     marker='o', markersize=8, linewidth=2.5)
    
    # Customize the plot
    plt.title(f"{ylabel} vs Noise Level", fontsize=14, pad=20)
    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend
    plt.legend(title="Model", title_fontsize=12, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to appropriate directory based on section
    if section == "accuracy":
        plt.savefig(os.path.join(accuracy_dir, filename), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(sustainability_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def line_plot_log(metric, ylabel, filename, section="accuracy"):
    if metric not in df.columns:
        print(f"⚠️ Skipping '{metric}' — not found in data.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the line plot with log scale
    ax = sns.lineplot(data=df, x="Noise_Level", y=metric, hue="Model", 
                     marker='o', markersize=8, linewidth=2.5)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Customize the plot
    plt.title(f"{ylabel} vs Noise Level (Log Scale)", fontsize=14, pad=20)
    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend
    plt.legend(title="Model", title_fontsize=12, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to appropriate directory based on section
    if section == "accuracy":
        plt.savefig(os.path.join(log_accuracy_dir, filename), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(log_sustainability_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# === Metrics to Plot ===

# H1: Model Performance (Accuracy Section)
line_plot("Accuracy", "Accuracy", "accuracy_vs_noise.png", "accuracy")
line_plot_log("Accuracy", "Accuracy", "accuracy_vs_noise.png", "accuracy")
line_plot("F1_Score", "F1 Score", "f1_score_vs_noise.png", "accuracy")
line_plot_log("F1_Score", "F1 Score", "f1_score_vs_noise.png", "accuracy")
line_plot("Recall", "Recall", "recall_vs_noise.png", "accuracy")
line_plot_log("Recall", "Recall", "recall_vs_noise.png", "accuracy")
line_plot("Precision", "Precision", "precision_vs_noise.png", "accuracy")
line_plot_log("Precision", "Precision", "precision_vs_noise.png", "accuracy")

# H2: Sustainability Metrics (Sustainability Section)
line_plot("CO2e_kg", "CO₂ Emissions (kg, CodeCarbon)", "co2_vs_noise.png", "sustainability")
line_plot_log("CO2e_kg", "CO₂ Emissions (kg, CodeCarbon)", "co2_vs_noise.png", "sustainability")
line_plot("Energy_kWh", "Energy (kWh, Intel Power Gadget)", "ipg_energy_vs_noise.png", "sustainability")
line_plot_log("Energy_kWh", "Energy (kWh, Intel Power Gadget)", "ipg_energy_vs_noise.png", "sustainability")
line_plot("Train_Time_sec", "Training Time (seconds)", "train_time_vs_noise.png", "sustainability")
line_plot_log("Train_Time_sec", "Training Time (seconds)", "train_time_vs_noise.png", "sustainability")
line_plot("Inference_Time_ms", "Inference Time (ms)", "inference_time_vs_noise.png", "sustainability")
line_plot_log("Inference_Time_ms", "Inference Time (ms)", "inference_time_vs_noise.png", "sustainability")
line_plot("RAM_MB", "Peak RAM Usage (MB)", "ram_usage_vs_noise.png", "sustainability")
line_plot_log("RAM_MB", "Peak RAM Usage (MB)", "ram_usage_vs_noise.png", "sustainability")
line_plot("Model_Size_MB", "Model Size (MB)", "model_size_vs_noise.png", "sustainability")
line_plot_log("Model_Size_MB", "Model Size (MB)", "model_size_vs_noise.png", "sustainability")

# H3: Tradeoff Metrics (Sustainability Section)
line_plot("Accuracy_per_CO2", "Accuracy per kg CO₂", "accuracy_per_co2_vs_noise.png", "sustainability")
line_plot_log("Accuracy_per_CO2", "Accuracy per kg CO₂", "accuracy_per_co2_vs_noise.png", "sustainability")
line_plot("F1_Score_per_CO2", "F1 Score per kg CO₂", "f1_per_co2_vs_noise.png", "sustainability")
line_plot_log("F1_Score_per_CO2", "F1 Score per kg CO₂", "f1_per_co2_vs_noise.png", "sustainability")
line_plot("Accuracy_per_Time", "Accuracy per second", "accuracy_per_time_vs_noise.png", "sustainability")
line_plot_log("Accuracy_per_Time", "Accuracy per second", "accuracy_per_time_vs_noise.png", "sustainability")
line_plot("Accuracy_per_RAM", "Accuracy per MB RAM", "accuracy_per_ram_vs_noise.png", "sustainability")
line_plot_log("Accuracy_per_RAM", "Accuracy per MB RAM", "accuracy_per_ram_vs_noise.png", "sustainability")

# === Save processed copy for reference ===
df.to_csv(os.path.join(output_dir, "processed_results.csv"), index=False)

# === Clean up duplicate files in main directory ===
# List of all graph files that should be in subdirectories
graph_files = [
    "accuracy_vs_noise.png",
    "f1_score_vs_noise.png",
    "recall_vs_noise.png",
    "precision_vs_noise.png",
    "co2_vs_noise.png",
    "ipg_energy_vs_noise.png",
    "train_time_vs_noise.png",
    "inference_time_vs_noise.png",
    "ram_usage_vs_noise.png",
    "model_size_vs_noise.png",
    "accuracy_per_co2_vs_noise.png",
    "f1_per_co2_vs_noise.png",
    "accuracy_per_time_vs_noise.png",
    "accuracy_per_ram_vs_noise.png"
]

# Remove duplicate files from main directory
for file in graph_files:
    file_path = os.path.join(output_dir, file)
    if os.path.exists(file_path):
        os.remove(file_path)

print("✓ Graphs generated and saved to:", output_dir)
print("✓ Duplicate files removed from main directory")
