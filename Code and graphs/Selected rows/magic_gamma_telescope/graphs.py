import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
def line_plot(metric, title, ylabel, section='accuracy'):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the line plot
    ax = sns.lineplot(data=df, x='Noise_Level', y=metric, hue='Model', 
                     marker='o', markersize=8, linewidth=2.5)
    
    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Noise Level (%)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend
    plt.legend(title='Model', title_fontsize=12, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot in the appropriate directory
    save_dir = accuracy_dir if section == 'accuracy' else sustainability_dir
    save_path = os.path.join(save_dir, f'{metric}_vs_noise.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def line_plot_log(metric, title, ylabel, section='accuracy'):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the line plot with log scale
    ax = sns.lineplot(data=df, x='Noise_Level', y=metric, hue='Model', 
                     marker='o', markersize=8, linewidth=2.5)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Noise Level (%)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend
    plt.legend(title='Model', title_fontsize=12, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot in the appropriate directory
    save_dir = log_accuracy_dir if section == 'accuracy' else log_sustainability_dir
    save_path = os.path.join(save_dir, f'{metric}_vs_noise.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# === Clean existing files ===
for dir_path in [accuracy_dir, sustainability_dir, log_accuracy_dir, log_sustainability_dir]:
    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            os.remove(os.path.join(dir_path, file))

# === Metrics to Plot ===

# Accuracy Metrics (Accuracy Section)
accuracy_metrics = [
    ("Accuracy", "Accuracy vs Noise Level", "Accuracy"),
    ("F1_Score", "F1 Score vs Noise Level", "F1 Score"),
    ("Recall", "Recall vs Noise Level", "Recall"),
    ("Precision", "Precision vs Noise Level", "Precision")
]

# Sustainability Metrics (Sustainability Section)
sustainability_metrics = [
    ("CO2e_kg", "CO₂ Emissions (kg) vs Noise Level", "CO₂ Emissions (kg, CodeCarbon)"),
    ("Energy_kWh", "Energy (kWh, Intel Power Gadget) vs Noise Level", "Energy (kWh, Intel Power Gadget)"),
    ("Train_Time_sec", "Training Time (seconds) vs Noise Level", "Training Time (seconds)"),
    ("Inference_Time_ms", "Inference Time (ms) vs Noise Level", "Inference Time (ms)"),
    ("RAM_MB", "Peak RAM Usage (MB) vs Noise Level", "Peak RAM Usage (MB)"),
    ("Model_Size_MB", "Model Size (MB) vs Noise Level", "Model Size (MB)"),
    ("Accuracy_per_CO2", "Accuracy per kg CO₂ vs Noise Level", "Accuracy per kg CO₂"),
    ("F1_Score_per_CO2", "F1 Score per kg CO₂ vs Noise Level", "F1 Score per kg CO₂"),
    ("Accuracy_per_Time", "Accuracy per second vs Noise Level", "Accuracy per second"),
    ("Accuracy_per_RAM", "Accuracy per MB RAM vs Noise Level", "Accuracy per MB RAM")
]

# Generate accuracy plots
for metric, title, ylabel in accuracy_metrics:
    line_plot(metric, title, ylabel, "accuracy")
    line_plot_log(metric, f"Log Scale: {title}", f"Log({ylabel})", "accuracy")

# Generate sustainability plots
for metric, title, ylabel in sustainability_metrics:
    line_plot(metric, title, ylabel, "sustainability")
    line_plot_log(metric, f"Log Scale: {title}", f"Log({ylabel})", "sustainability")

# === Save processed copy for reference ===
df.to_csv(os.path.join(output_dir, "processed_results.csv"), index=False)

print("✓ Graphs generated and saved to:", output_dir, "and", log_output_dir)
print("✓ Graphs organized into accuracy/ and sustainability/ subdirectories")
