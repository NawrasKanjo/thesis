import os
import time
import joblib
import tracemalloc
import subprocess
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from codecarbon import EmissionsTracker
from sklearn.preprocessing import StandardScaler


# === Config ===
FEATURES = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABEL = "activity"
noise_levels = list(range(0, 101, 10))
seed = 42

dataset_dir = "../../../../Datasets/Selected rows/walk_run/noisy_datasets"
results_dir = "results_corrected"
log_dir_root = "logs"
carbon_dir = "codecarbon_logs"
temp_dir = "temp"
ipg_path = r"C:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe"

for d in [results_dir, log_dir_root, carbon_dir, temp_dir]:
    os.makedirs(d, exist_ok=True)

# === Models ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}
scaled_models = ["LogisticRegression", "SVM", "KNN"]

def read_energy_kwh(ipg_log_path):
    try:
        df = pd.read_csv(ipg_log_path)
        col = "Cumulative Processor Energy_0(Joules)"
        if col in df.columns:
            return (df[col].dropna().iloc[-1] - df[col].dropna().iloc[0]) / 3.6e6
    except Exception as e:
        print(f"⚠️ IPG read failed: {e}")
    return None

results = []

for noise in noise_levels:
    df = pd.read_csv(os.path.join(dataset_dir, f"dataset_{noise}.csv"))
    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=seed)

    for model_name, model_template in models.items():
        model = copy.deepcopy(model_template)

        ipg_log_dir = os.path.join(log_dir_root, model_name, f"noise_{noise}")
        os.makedirs(ipg_log_dir, exist_ok=True)
        ipg_log_path = os.path.join(ipg_log_dir, "intel_power_gadget_log.csv")

        try:
            ipg_proc = subprocess.Popen([
                ipg_path,
                "-resolution", "1000", "-duration", "100", "-file", ipg_log_path
            ])
        except FileNotFoundError:
            print("⚠️ Intel Power Gadget not found. Skipping energy logging.")
            ipg_proc = None

        tracker = EmissionsTracker(
            output_dir=carbon_dir,
            measure_power_secs=1,
            log_level="error"
        )
        tracker.start()
        tracemalloc.start()
        start_train = time.time()

        X_train_used = X_train
        X_test_used = X_test
        if model_name in scaled_models:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)

        model.fit(X_train_used, y_train)


        train_time = time.time() - start_train
        emissions = tracker.stop()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Inference
        start_pred = time.time()
        y_pred = model.predict(X_test_used)
        inference_time = (time.time() - start_pred) / len(X_test)

        # Save model & size
        model_path = os.path.join(temp_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
        os.remove(model_path)

        if ipg_proc:
            ipg_proc.wait(timeout=120)
        ipg_energy = read_energy_kwh(ipg_log_path) if ipg_proc else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)

        results.append({
            "Model": model_name,
            "Noise_Level": noise,
            "Accuracy": acc,
            "F1_Score": f1,
            "Recall": recall,
            "Precision": precision,
            "CO2e_kg": emissions,
            "Energy_kWh": ipg_energy,
            "Train_Time_sec": train_time,
            "Inference_Time_ms": inference_time * 1000,
            "RAM_MB": peak_mem / (1024 ** 2),
            "Model_Size_MB": model_size_mb,
            "Accuracy_per_CO2": acc / emissions if emissions else None,
            "F1_Score_per_CO2": f1 / emissions if emissions else None,
            "Accuracy_per_Time": acc / train_time if train_time else None,
            "Accuracy_per_RAM": acc / (peak_mem / (1024 ** 2)) if peak_mem else None
        })

        ipg_str = f"{ipg_energy:.6f} kWh" if ipg_energy is not None else "N/A"
        print(f"[{model_name}] Noise {noise}% → Acc={acc:.6f} | F1={f1:.6f} | CO2e={emissions:.9f} | IPG={ipg_str}")

# Save results
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(results_dir, "model_performance_results.csv"), index=False)
print("✓ All results saved.")
