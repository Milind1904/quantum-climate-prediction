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
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("QUANTUM VQC MODEL: Advanced Variational Quantum Classifier")
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

n_qubits = 6  # Increase to 6 qubits for better representation
pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X_scaled.shape[1]}")
print(f"Quantum qubits: {n_qubits}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================

print("\n[STEP 4] Train-Test Split...")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Use stratified sampling - balance between natural and balanced
# 75-25 split for better minority class learning
max_train = 500  # Manageable size for VQC
n_train_0 = int(max_train * 0.75)
n_train_1 = max_train - n_train_0

np.random.seed(42)
idx_train_0 = np.where(y_train == 0)[0]
idx_train_1 = np.where(y_train == 1)[0]

selected_train_0 = np.random.choice(idx_train_0, min(n_train_0, len(idx_train_0)), replace=False)
selected_train_1 = np.random.choice(idx_train_1, min(n_train_1, len(idx_train_1)), replace=False)

train_indices = np.concatenate([selected_train_0, selected_train_1])
np.random.shuffle(train_indices)

X_train_quantum = X_train[train_indices]
y_train_quantum = y_train[train_indices]

print(f"Training samples: {len(train_indices)} (75-25 distribution)")
print(f"  Class 0: {np.sum(y_train_quantum == 0)}")
print(f"  Class 1: {np.sum(y_train_quantum == 1)}")

# Test set
max_test = 200
n_test_0 = int(max_test * 0.75)
n_test_1 = max_test - n_test_0

idx_test_0 = np.where(y_test == 0)[0]
idx_test_1 = np.where(y_test == 1)[0]

selected_test_0 = np.random.choice(idx_test_0, min(n_test_0, len(idx_test_0)), replace=False)
selected_test_1 = np.random.choice(idx_test_1, min(n_test_1, len(idx_test_1)), replace=False)

test_indices = np.concatenate([selected_test_0, selected_test_1])
np.random.shuffle(test_indices)

X_test_quantum = X_test[test_indices]
y_test_quantum = y_test[test_indices]

print(f"Test samples: {len(test_indices)} (75-25 distribution)")
print(f"  Class 0: {np.sum(y_test_quantum == 0)}")
print(f"  Class 1: {np.sum(y_test_quantum == 1)}")

# ============================================================================
# STEP 5: BUILD VQC (Variational Quantum Classifier)
# ============================================================================

print("\n[STEP 5] Building Variational Quantum Classifier...")
print("-"*70)

# Feature map
feature_map = PauliFeatureMap(
    feature_dimension=n_qubits,
    reps=2,
    paulis=['Z', 'ZZ'],
    entanglement='full'
)

print(f"Feature Map: PauliFeatureMap")
print(f"  - Qubits: {n_qubits}")
print(f"  - Repetitions: 2")
print(f"  - Paulis: Z, ZZ")
print(f"  - Entanglement: full")

# Variational form (ansatz)
ansatz = RealAmplitudes(
    num_qubits=n_qubits,
    reps=3,
    entanglement='full'
)

print(f"\nAnsatz: RealAmplitudes")
print(f"  - Qubits: {n_qubits}")
print(f"  - Repetitions: 3")
print(f"  - Entanglement: full")

# Backend
sampler = Sampler()
print(f"\nSampler: Qiskit Sampler")

# Optimizer
optimizer = COBYLA(maxiter=100)
print(f"Optimizer: COBYLA (100 iterations)")

# ============================================================================
# STEP 6: TRAIN VQC
# ============================================================================

print("\n[STEP 6] Training Variational Quantum Classifier...")
print("-"*70)
print("⏳ This will take several minutes...")

start_time = time.time()

algorithm_globals.random_seed = 42

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

print("Fitting VQC model...")
vqc.fit(X_train_quantum, y_train_quantum)

training_time = time.time() - start_time
print(f"✓ VQC training completed in {training_time:.2f} seconds")

# ============================================================================
# STEP 7: EVALUATE VQC
# ============================================================================

print("\n[STEP 7] Evaluating VQC Model...")
print("-"*70)

y_pred_quantum = vqc.predict(X_test_quantum)

print(f"\nPredictions distribution:")
print(f"  Predicted class 0: {np.sum(y_pred_quantum == 0)}")
print(f"  Predicted class 1: {np.sum(y_pred_quantum == 1)}")

# Metrics
quantum_accuracy = accuracy_score(y_test_quantum, y_pred_quantum)
quantum_precision = precision_score(y_test_quantum, y_pred_quantum, zero_division=0)
quantum_recall = recall_score(y_test_quantum, y_pred_quantum, zero_division=0)
quantum_f1 = f1_score(y_test_quantum, y_pred_quantum, zero_division=0)

print(f"\n{'Metric':<20} {'Score':<15}")
print("-"*35)
print(f"{'Accuracy':<20} {quantum_accuracy:.4f}")
print(f"{'Precision':<20} {quantum_precision:.4f}")
print(f"{'Recall':<20} {quantum_recall:.4f}")
print(f"{'F1-Score':<20} {quantum_f1:.4f}")

cm_quantum = confusion_matrix(y_test_quantum, y_pred_quantum)
print(f"\nConfusion Matrix:\n{cm_quantum}")

print("\nClassification Report:")
print(classification_report(y_test_quantum, y_pred_quantum))

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n[STEP 8] Saving Results...")
print("-"*70)

quantum_results = {
    'model_type': 'VQC',
    'accuracy': float(quantum_accuracy),
    'precision': float(quantum_precision),
    'recall': float(quantum_recall),
    'f1_score': float(quantum_f1),
    'training_time': float(training_time),
    'qubits': int(n_qubits),
    'feature_map': 'PauliFeatureMap',
    'feature_map_reps': 2,
    'ansatz': 'RealAmplitudes',
    'ansatz_reps': 3,
    'optimizer': 'COBYLA',
    'optimizer_maxiter': 100,
    'explained_variance': float(pca.explained_variance_ratio_.sum()),
    'training_samples': int(len(y_train_quantum)),
    'test_samples': int(len(y_test_quantum))
}

with open('quantum_vqc_results.json', 'w') as f:
    json.dump(quantum_results, f, indent=4)

print("✓ Results saved to quantum_vqc_results.json")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================

print("\n[STEP 9] Creating Visualizations...")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_quantum, display_labels=['Normal', 'Extreme'])
cm_display.plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('VQC - Confusion Matrix', fontsize=12, fontweight='bold')

# Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [quantum_accuracy, quantum_precision, quantum_recall, quantum_f1]
bars = axes[0, 1].bar(metrics_names, metrics_values, color='darkblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('VQC - Performance Metrics', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 1.0])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Model Architecture
arch_text = f"""VQC Architecture:
• Feature Map: PauliFeatureMap
• Qubits: {n_qubits}
• Feature Reps: 2
• Paulis: Z, ZZ
• Ansatz: RealAmplitudes
• Ansatz Reps: 3
• Entanglement: Full
• Optimizer: COBYLA (100 iter)
• Training Time: {training_time:.1f}s
"""
axes[1, 0].text(0.1, 0.5, arch_text, fontsize=11, verticalalignment='center', 
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
axes[1, 0].axis('off')
axes[1, 0].set_title('VQC Architecture', fontsize=12, fontweight='bold')

# PCA Explained Variance
axes[1, 1].plot(np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2, markersize=8, color='darkblue')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Explained Variance')
axes[1, 1].set_title('PCA - Explained Variance', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('quantum_vqc_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: quantum_vqc_results.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("VQC MODEL TRAINING COMPLETE")
print("="*70)
print(f"""
📊 VQC Results:
   Accuracy:  {quantum_accuracy:.4f}
   Precision: {quantum_precision:.4f}
   Recall:    {quantum_recall:.4f}
   F1-Score:  {quantum_f1:.4f}
   
⏱️  Training Time: {training_time:.2f} seconds
   
🔬 Model Details:
   Qubits: {n_qubits}
   Feature Map: PauliFeatureMap (2 reps)
   Ansatz: RealAmplitudes (3 reps)
   Optimizer: COBYLA (100 iterations)
   Explained Variance: {pca.explained_variance_ratio_.sum():.4f}

📁 Output Files:
   ✓ quantum_vqc_results.json
   ✓ quantum_vqc_results.png
""")

print("✓ VQC model training complete!")
