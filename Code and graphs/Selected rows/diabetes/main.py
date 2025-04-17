import os
import time
import joblib
import tracemalloc
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
import copy

# --- Model Definitions ---
models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

# --- Configuration ---
dataset_dir = "../../../../Datasets/Selected rows/diabetes/noisy_datasets"
results_dir = "results_corrected"
temp_dir = "temp"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
noise_levels = list(range(0, 101, 10))
FEATURES = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
seed = 42

# --- Output container ---
results = []

# --- Main Loop ---
for noise in noise_levels:
    file_path = os.path.join(dataset_dir, f"dataset_{noise}.csv")
    df = pd.read_csv(file_path)
    X = df[FEATURES]
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Pre-scale
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name, model_proto in models.items():
        model = copy.deepcopy(model_proto)

        # Select whether to use scaled or raw features
        if model_name in ["SVM", "LogisticRegression", "KNN"]:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled
        else:
            X_train_used = X_train
            X_test_used = X_test

        # Resource tracking
        tracker = EmissionsTracker(measure_power_secs=1, output_dir=temp_dir, log_level="error")
        tracker.start()
        tracemalloc.start()
        start_train = time.time()

        model.fit(X_train_used, y_train)

        train_time = time.time() - start_train
        emissions = tracker.stop()

        # Inference time
        start_pred = time.time()
        y_pred = model.predict(X_test_used)
        inference_time = (time.time() - start_pred) / len(X_test)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_usage_mb = peak_mem / (1024 * 1024)

        # Model size
        model_path = os.path.join(temp_dir, f"{model_name}_temp.joblib")
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)

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
            "Train_Time_sec": train_time,
            "Inference_Time_ms": inference_time * 1000,
            "RAM_MB": ram_usage_mb,
            "Model_Size_MB": model_size_mb,
            "Accuracy_per_CO2": acc / emissions if emissions > 0 else None,
            "F1_Score_per_CO2": f1 / emissions if emissions > 0 else None,
            "Accuracy_per_Time": acc / train_time if train_time > 0 else None,
            "Accuracy_per_RAM": acc / ram_usage_mb if ram_usage_mb > 0 else None
        })

        print(f"[{model_name}] Noise {noise}% → Acc: {acc:.3f} | F1: {f1:.3f} | CO2e: {emissions:.6f} kg")

# --- Save Results ---
results_df = pd.DataFrame(results)
results_path = os.path.join(results_dir, "model_performance_results_corrected.csv")
results_df.to_csv(results_path, index=False)
print(f"✓ All results saved to {results_path}")

