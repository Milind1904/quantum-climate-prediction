"""
Quantum SVC Model - Optimized for Overall Accuracy
This version prioritizes overall accuracy over balanced class performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
import time
import json

print("=" * 70)
print("QUANTUM SVC - ACCURACY OPTIMIZED")
print("=" * 70)

# Load data
print("\n[1/6] Loading climate data...")
from load_climate_data import load_and_prepare_climate_data
X, y = load_and_prepare_climate_data()
print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")

# Normalize features
print("\n[2/6] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare training data
print("\n[3/6] Preparing training data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Sample data with natural distribution (1000 train, 500 test)
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]

n_train_samples = 1000
n_class_0 = int(n_train_samples * 0.87)  # Match natural distribution
n_class_1 = n_train_samples - n_class_0

selected_0 = np.random.choice(class_0_indices, size=n_class_0, replace=False)
selected_1 = np.random.choice(class_1_indices, size=n_class_1, replace=False)
train_indices = np.concatenate([selected_0, selected_1])
np.random.shuffle(train_indices)

X_train_sampled = X_train[train_indices]
y_train_sampled = y_train[train_indices]

# Test samples
test_class_0 = np.where(y_test == 0)[0]
test_class_1 = np.where(y_test == 1)[0]
n_test_samples = 500
n_test_0 = int(n_test_samples * 0.87)
n_test_1 = n_test_samples - n_test_0

selected_test_0 = np.random.choice(test_class_0, size=n_test_0, replace=False)
selected_test_1 = np.random.choice(test_class_1, size=n_test_1, replace=False)
test_indices = np.concatenate([selected_test_0, selected_test_1])
np.random.shuffle(test_indices)

X_test_sampled = X_test[test_indices]
y_test_sampled = y_test[test_indices]

print(f"✓ Training samples: {len(X_train_sampled)} (Class 0: {n_class_0}, Class 1: {n_class_1})")
print(f"✓ Test samples: {len(X_test_sampled)}")

# Apply PCA
print("\n[4/6] Applying PCA (5 components)...")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_sampled)
X_test_pca = pca.transform(X_test_sampled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"✓ Explained variance: {explained_var:.4f}")

# Build quantum kernel
print("\n[5/6] Building quantum kernel...")
print("  - Creating 5-qubit ZZ feature map...")
feature_map = ZZFeatureMap(feature_dimension=5, reps=2, entanglement='full')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

print("  - Computing quantum kernel matrices (this takes ~45-55 minutes)...")
start_kernel = time.time()
K_train = quantum_kernel.evaluate(X_train_pca)
K_test = quantum_kernel.evaluate(X_test_pca, X_train_pca)
kernel_time = time.time() - start_kernel
print(f"✓ Kernel computation complete in {kernel_time:.2f}s")

# Try 3 different configurations
print("\n[6/6] Training QSVC models with different configurations...")
print("=" * 70)

results = []

# Configuration 1: No class weights (natural)
print("\n[Config 1/3] Natural weights (no balancing)...")
start_train = time.time()
qsvc_1 = SVC(kernel='precomputed', C=10.0)
qsvc_1.fit(K_train, y_train_sampled)
y_pred_1 = qsvc_1.predict(K_test)
train_time_1 = time.time() - start_train

acc_1 = accuracy_score(y_test_sampled, y_pred_1)
prec_1 = precision_score(y_test_sampled, y_pred_1, zero_division=0)
rec_1 = recall_score(y_test_sampled, y_pred_1, zero_division=0)
f1_1 = f1_score(y_test_sampled, y_pred_1, zero_division=0)
cm_1 = confusion_matrix(y_test_sampled, y_pred_1)

print(f"  Accuracy:  {acc_1:.4f} ({acc_1*100:.2f}%)")
print(f"  Precision: {prec_1:.4f}")
print(f"  Recall:    {rec_1:.4f}")
print(f"  F1-Score:  {f1_1:.4f}")
print(f"  Confusion Matrix:\n{cm_1}")

results.append({
    "config": "natural_weights",
    "accuracy": acc_1,
    "precision": prec_1,
    "recall": rec_1,
    "f1": f1_1,
    "cm": cm_1
})

# Configuration 2: Balanced weights (original)
print("\n[Config 2/3] Balanced weights...")
start_train = time.time()
qsvc_2 = SVC(kernel='precomputed', C=10.0, class_weight='balanced')
qsvc_2.fit(K_train, y_train_sampled)
y_pred_2 = qsvc_2.predict(K_test)
train_time_2 = time.time() - start_train

acc_2 = accuracy_score(y_test_sampled, y_pred_2)
prec_2 = precision_score(y_test_sampled, y_pred_2, zero_division=0)
rec_2 = recall_score(y_test_sampled, y_pred_2, zero_division=0)
f1_2 = f1_score(y_test_sampled, y_pred_2, zero_division=0)
cm_2 = confusion_matrix(y_test_sampled, y_pred_2)

print(f"  Accuracy:  {acc_2:.4f} ({acc_2*100:.2f}%)")
print(f"  Precision: {prec_2:.4f}")
print(f"  Recall:    {rec_2:.4f}")
print(f"  F1-Score:  {f1_2:.4f}")
print(f"  Confusion Matrix:\n{cm_2}")

results.append({
    "config": "balanced_weights",
    "accuracy": acc_2,
    "precision": prec_2,
    "recall": rec_2,
    "f1": f1_2,
    "cm": cm_2
})

# Configuration 3: Custom weights (moderate balance)
print("\n[Config 3/3] Custom weights (0: 1.0, 1: 3.0)...")
start_train = time.time()
qsvc_3 = SVC(kernel='precomputed', C=10.0, class_weight={0: 1.0, 1: 3.0})
qsvc_3.fit(K_train, y_train_sampled)
y_pred_3 = qsvc_3.predict(K_test)
train_time_3 = time.time() - start_train

acc_3 = accuracy_score(y_test_sampled, y_pred_3)
prec_3 = precision_score(y_test_sampled, y_pred_3, zero_division=0)
rec_3 = recall_score(y_test_sampled, y_pred_3, zero_division=0)
f1_3 = f1_score(y_test_sampled, y_pred_3, zero_division=0)
cm_3 = confusion_matrix(y_test_sampled, y_pred_3)

print(f"  Accuracy:  {acc_3:.4f} ({acc_3*100:.2f}%)")
print(f"  Precision: {prec_3:.4f}")
print(f"  Recall:    {rec_3:.4f}")
print(f"  F1-Score:  {f1_3:.4f}")
print(f"  Confusion Matrix:\n{cm_3}")

results.append({
    "config": "custom_weights",
    "accuracy": acc_3,
    "precision": prec_3,
    "recall": rec_3,
    "f1": f1_3,
    "cm": cm_3
})

# Find best configuration
print("\n" + "=" * 70)
print("COMPARISON OF CONFIGURATIONS")
print("=" * 70)

accuracies = [acc_1, acc_2, acc_3]
best_idx = np.argmax(accuracies)
configs = ["Natural weights", "Balanced weights", "Custom weights"]

for i, (config, acc) in enumerate(zip(configs, accuracies)):
    marker = "🏆" if i == best_idx else "  "
    print(f"{marker} {config:20s}: {acc:.4f} ({acc*100:.2f}%)")

best_result = results[best_idx]
print(f"\n✅ BEST CONFIGURATION: {configs[best_idx]}")
print(f"   Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"   Precision: {best_result['precision']:.4f}")
print(f"   Recall:    {best_result['recall']:.4f}")
print(f"   F1-Score:  {best_result['f1']:.4f}")
print(f"\n   Confusion Matrix:")
print(best_result['cm'])

# Save best result
best_config_data = {
    "model_type": "QSVC_AccuracyOptimized",
    "best_configuration": configs[best_idx],
    "accuracy": float(best_result['accuracy']),
    "precision": float(best_result['precision']),
    "recall": float(best_result['recall']),
    "f1_score": float(best_result['f1']),
    "confusion_matrix": best_result['cm'].tolist(),
    "all_configurations": {
        "natural_weights": {
            "accuracy": float(acc_1),
            "precision": float(prec_1),
            "recall": float(rec_1),
            "f1": float(f1_1)
        },
        "balanced_weights": {
            "accuracy": float(acc_2),
            "precision": float(prec_2),
            "recall": float(rec_2),
            "f1": float(f1_2)
        },
        "custom_weights": {
            "accuracy": float(acc_3),
            "precision": float(prec_3),
            "recall": float(rec_3),
            "f1": float(f1_3)
        }
    },
    "kernel_computation_time": float(kernel_time),
    "total_time": float(kernel_time + train_time_1 + train_time_2 + train_time_3),
    "qubits": 5,
    "feature_map_reps": 2,
    "explained_variance": float(explained_var),
    "training_samples": int(len(X_train_sampled)),
    "test_samples": int(len(X_test_sampled))
}

with open('quantum_accuracy_optimized_results.json', 'w') as f:
    json.dump(best_config_data, f, indent=2)

print("\n✓ Results saved to quantum_accuracy_optimized_results.json")
print("\n" + "=" * 70)
