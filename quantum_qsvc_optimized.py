import numpy as np
import pandas as pd
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# Quantum imports
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.utils import algorithm_globals
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("QUANTUM MODEL (OPTIMIZED): QSVC with Better Sampling Strategy")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print("\n[STEP 1] Loading Dataset...")
print("-"*70)

from load_climate_data import load_and_prepare_climate_data
X, y = load_and_prepare_climate_data()

if X is None:
    print("ERROR: Failed to load dataset!")
    exit()

print(f"\n✓ Dataset loaded successfully")
print(f"✓ Features shape: {X.shape}")
print(f"✓ Labels shape: {y.shape}")
print(f"✓ Class distribution: {np.bincount(y)}")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

print("\n[STEP 2] Data Preprocessing...")
print("-"*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features standardized")

# ============================================================================
# STEP 3: DIMENSIONALITY REDUCTION
# ============================================================================

print("\n[STEP 3] Dimensionality Reduction for Quantum...")
print("-"*70)

n_qubits = 5
pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X_scaled.shape[1]}")
print(f"Quantum qubits: {n_qubits}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT - KEEP NATURAL DISTRIBUTION
# ============================================================================

print("\n[STEP 4] Train-Test Split (Natural Distribution)...")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keep natural 87-13 split
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print(f"Training labels: {np.bincount(y_train)}")
print(f"Testing labels: {np.bincount(y_test)}")

# ============================================================================
# STEP 5: QUANTUM CIRCUIT SETUP
# ============================================================================

print("\n[STEP 5] Setting up Quantum Circuit...")
print("-"*70)

feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=2,  # 2 reps for good accuracy
    entanglement='full'
)

print(f"Feature map: ZZFeatureMap")
print(f"  - Qubits: {n_qubits}")
print(f"  - Repetitions: 2")
print(f"  - Entanglement: full")

backend = AerSimulator()
print(f"Backend: Aer Simulator")

# ============================================================================
# STEP 6: CREATE QUANTUM KERNEL
# ============================================================================

print("\n[STEP 6] Creating Quantum Kernel...")
print("-"*70)
print("⏳ This may take a few minutes...")

start_time = time.time()

algorithm_globals.random_seed = 42

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

# Use stratified sampling - keep natural distribution
print("Computing kernel matrix for training data...")
max_samples = 1000
np.random.seed(42)

# Sample proportionally: 87% class 0, 13% class 1 (natural distribution)
n_class_0 = int(max_samples * 0.87)
n_class_1 = max_samples - n_class_0

idx_class_0 = np.where(y_train == 0)[0]
idx_class_1 = np.where(y_train == 1)[0]

# Random sample
selected_idx_0 = np.random.choice(idx_class_0, min(n_class_0, len(idx_class_0)), replace=False)
selected_idx_1 = np.random.choice(idx_class_1, min(n_class_1, len(idx_class_1)), replace=False)

train_indices = np.concatenate([selected_idx_0, selected_idx_1])
np.random.shuffle(train_indices)

X_train_quantum = X_train[train_indices]
y_train_quantum = y_train[train_indices]

print(f"Using {len(train_indices)} samples (natural distribution 87-13 for high accuracy)")
print(f"  Class 0: {np.sum(y_train_quantum == 0)} ({np.sum(y_train_quantum == 0)/len(y_train_quantum)*100:.1f}%)")
print(f"  Class 1: {np.sum(y_train_quantum == 1)} ({np.sum(y_train_quantum == 1)/len(y_train_quantum)*100:.1f}%)")

kernel_matrix = quantum_kernel.evaluate(X_train_quantum)

kernel_time = time.time() - start_time
print(f"✓ Kernel computed in {kernel_time:.2f} seconds")
print(f"Kernel matrix shape: {kernel_matrix.shape}")

# ============================================================================
# STEP 7: TRAIN QSVC
# ============================================================================

print("\n[STEP 7] Training Quantum Support Vector Classifier...")
print("-"*70)

start_time = time.time()

qsvc = SVC(kernel='precomputed', C=10.0, class_weight='balanced')
print("Fitting QSVC with quantum kernel...")
qsvc.fit(kernel_matrix, y_train_quantum)

training_time = time.time() - start_time
print(f"✓ QSVC training completed in {training_time:.2f} seconds")
print(f"Support vectors: {qsvc.n_support_}")

# ============================================================================
# STEP 8: EVALUATE - USE NATURAL DISTRIBUTION
# ============================================================================

print("\n[STEP 8] Evaluating Quantum Model...")
print("-"*70)

# Sample test set with natural distribution
max_test = 500
n_test_0 = int(max_test * 0.87)
n_test_1 = max_test - n_test_0

idx_test_0 = np.where(y_test == 0)[0]
idx_test_1 = np.where(y_test == 1)[0]

selected_test_0 = np.random.choice(idx_test_0, min(n_test_0, len(idx_test_0)), replace=False)
selected_test_1 = np.random.choice(idx_test_1, min(n_test_1, len(idx_test_1)), replace=False)

test_indices = np.concatenate([selected_test_0, selected_test_1])
np.random.shuffle(test_indices)

X_test_quantum = X_test[test_indices]
y_test_quantum = y_test[test_indices]

print(f"Computing kernel matrix for test data...")
print(f"Using {len(test_indices)} test samples (natural distribution)")
print(f"  Class 0: {np.sum(y_test_quantum == 0)} ({np.sum(y_test_quantum == 0)/len(y_test_quantum)*100:.1f}%)")
print(f"  Class 1: {np.sum(y_test_quantum == 1)} ({np.sum(y_test_quantum == 1)/len(y_test_quantum)*100:.1f}%)")

kernel_test = quantum_kernel.evaluate(X_test_quantum, X_train_quantum)
y_pred_quantum = qsvc.predict(kernel_test)

# Get decision function scores for ROC/PR curves
y_scores_quantum = qsvc.decision_function(kernel_test)

print(f"\nPredictions distribution:")
print(f"  Predicted class 0: {np.sum(y_pred_quantum == 0)}")
print(f"  Predicted class 1: {np.sum(y_pred_quantum == 1)}")

# Metrics - Overall accuracy + Class 0 (Normal Weather) specific metrics
quantum_accuracy = accuracy_score(y_test_quantum, y_pred_quantum)

# Calculate confusion matrix
cm_quantum = confusion_matrix(y_test_quantum, y_pred_quantum)
tn, fp, fn, tp = cm_quantum[0, 0], cm_quantum[0, 1], cm_quantum[1, 0], cm_quantum[1, 1]

# Class 0 (Normal Weather) metrics only
quantum_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for class 0
quantum_recall = tn / (tn + fp) if (tn + fp) > 0 else 0     # Recall for class 0
quantum_f1 = 2 * (quantum_precision * quantum_recall) / (quantum_precision + quantum_recall) if (quantum_precision + quantum_recall) > 0 else 0

print(f"\n{'Metric':<20} {'Score':<15}")
print("-"*35)
print(f"{'Accuracy (Overall)':<20} {quantum_accuracy:.4f}")
print(f"{'Precision (Class 0)':<20} {quantum_precision:.4f}")
print(f"{'Recall (Class 0)':<20} {quantum_recall:.4f}")
print(f"{'F1-Score (Class 0)':<20} {quantum_f1:.4f}")

print(f"\nConfusion Matrix:\n{cm_quantum}")
print(f"\nNote: Precision, Recall, and F1-Score are for Normal Weather (Class 0) only")

print("\nClassification Report:")
print(classification_report(y_test_quantum, y_pred_quantum))

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n[STEP 9] Saving Results...")
print("-"*70)

quantum_results = {
    'model_type': 'QSVC_Optimized',
    'accuracy': float(quantum_accuracy),
    'precision': float(quantum_precision),  # Class 0 (Normal) only
    'recall': float(quantum_recall),        # Class 0 (Normal) only
    'f1_score': float(quantum_f1),          # Class 0 (Normal) only
    'metrics_note': 'Precision, Recall, and F1 are for Normal Weather (Class 0) only',
    'kernel_computation_time': float(kernel_time),
    'training_time': float(training_time),
    'total_time': float(kernel_time + training_time),
    'qubits': int(n_qubits),
    'feature_map_reps': 2,
    'explained_variance': float(pca.explained_variance_ratio_.sum()),
    'training_samples': int(len(y_train_quantum)),
    'test_samples': int(len(y_test_quantum)),
    'C_parameter': 10.0,
    'sampling_strategy': 'natural_distribution_87_13',
    'confusion_matrix': cm_quantum.tolist()
}

with open('quantum_results.json', 'w') as f:
    json.dump(quantum_results, f, indent=4)

print("✓ Results saved to quantum_results.json")

# Save predictions for ROC/PR curves
np.savez('quantum_predictions.npz',
         y_true=y_test_quantum,
         y_pred=y_pred_quantum,
         y_scores=y_scores_quantum)
print("✓ Predictions saved to quantum_predictions.npz")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================

print("\n[STEP 10] Creating Visualizations...")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_quantum, display_labels=['Normal', 'Extreme'])
cm_display.plot(ax=axes[0, 0], cmap='Greens')
axes[0, 0].set_title('QSVC - Confusion Matrix', fontsize=12, fontweight='bold')

# Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [quantum_accuracy, quantum_precision, quantum_recall, quantum_f1]
bars = axes[0, 1].bar(metrics_names, metrics_values, color='darkgreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('QSVC - Performance Metrics', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 1.0])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Timing Breakdown
timing_labels = ['Kernel\nComputation', 'QSVC\nTraining', 'Total']
timing_values = [kernel_time, training_time, kernel_time + training_time]
axes[1, 0].bar(timing_labels, timing_values, color='darkgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Time (seconds)')
axes[1, 0].set_title('QSVC - Timing Breakdown', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(timing_values):
    axes[1, 0].text(i, v + max(timing_values)*0.02, f'{v:.1f}s', ha='center', fontweight='bold')

# PCA Explained Variance
axes[1, 1].plot(np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2, markersize=8, color='darkgreen')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Explained Variance')
axes[1, 1].set_title('PCA - Explained Variance', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('quantum_qsvc_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: quantum_qsvc_results.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("QUANTUM MODEL TRAINING COMPLETE")
print("="*70)
print(f"""
📊 QSVC Results (Optimized):
   Accuracy:  {quantum_accuracy:.4f}
   Precision: {quantum_precision:.4f}
   Recall:    {quantum_recall:.4f}
   F1-Score:  {quantum_f1:.4f}
   
⏱️  Quantum Kernel Time: {kernel_time:.2f} seconds
⏱️  QSVC Training Time: {training_time:.2f} seconds
⏱️  Total Time: {kernel_time + training_time:.2f} seconds
   
🔬 Model Details:
   Qubits (Features): {n_qubits}
   Feature Map Reps: 2
   Explained Variance: {pca.explained_variance_ratio_.sum():.4f}
   Sampling: Natural Distribution (87% normal, 13% extreme) for realistic evaluation

📁 Output Files:
   ✓ quantum_results.json
   ✓ quantum_qsvc_results.png
""")

print("✓ Quantum model training complete!")
