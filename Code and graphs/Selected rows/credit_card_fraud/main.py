import os
import time
import joblib
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from codecarbon import EmissionsTracker

# ---------------------
# MODEL DEFINITIONS
# ---------------------
models = {
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

# ---------------------
# CONFIGURATION
# ---------------------
dataset_dir = "../../../../Datasets/Selected rows/credit_card_fraud/noisy_datasets"
results_dir = "results_corrected"
temp_dir = "temp"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

sensor_cols = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

noise_levels = list(range(0, 101, 10))
seed = 42
results = []

# ---------------------
# EXPERIMENT LOOP
# ---------------------
for noise in noise_levels:
    file_path = os.path.join(dataset_dir, f"dataset_{noise}.csv")
    df = pd.read_csv(file_path)

    X = df[sensor_cols]
    y = df["Class"]  # 0 = legitimate, 1 = fraud

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    for name, base_model in models.items():
        model = base_model.__class__(**base_model.get_params())

        tracker = EmissionsTracker(project_name=f"{name}_noise_{noise}", output_dir=temp_dir, log_level="error")
        tracker.start()
        tracemalloc.start()
        start_train = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - start_train
        emissions = tracker.stop()

        start_pred = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_pred) / len(X_test)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_usage_mb = peak_mem / (1024 * 1024)

        model_path = os.path.join(temp_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)

        results.append({
            "Model": name,
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

        print(f"[{name}] Noise {noise}% | Acc: {acc:.3f} | F1: {f1:.3f} | CO2e: {emissions:.6f} kg")

# ---------------------
# SAVE RESULTS
# ---------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, "model_performance_results_corrected.csv"), index=False)
print("✓ All results saved.")
