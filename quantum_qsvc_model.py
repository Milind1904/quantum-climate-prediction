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
print("QUANTUM MODEL: QSVC (Quantum Support Vector Classifier)")
print("="*70)

# ============================================================================
# STEP 1: LOAD YOUR DATASET
# ============================================================================

print("\n[STEP 1] Loading Dataset...")
print("-"*70)

# Import data loader
from load_climate_data import load_and_prepare_climate_data

# Load preprocessed data
X, y = load_and_prepare_climate_data()

if X is None:
    print("ERROR: Failed to load dataset!")
    exit()

print(f"\n✓ Dataset loaded successfully")
print(f"✓ Features shape: {X.shape}")
print(f"✓ Labels shape: {y.shape}")
print(f"✓ Class distribution: {np.bincount(y)}")

# ============================================================================
# STEP 2: DATA PREPROCESSING & FEATURE SELECTION
# ============================================================================

print("\n[STEP 2] Data Preprocessing...")
print("-"*70)

# Select most important features using variance and correlation
from sklearn.feature_selection import SelectKBest, f_classif

# First normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select top features based on ANOVA F-value
selector = SelectKBest(f_classif, k=10)  # Select top 10 most discriminative features
X_selected = selector.fit_transform(X_scaled, y)

print("✓ Features standardized and selected")
print(f"Selected {X_selected.shape[1]} most discriminative features")

# ============================================================================
# STEP 3: DIMENSIONALITY REDUCTION FOR QUANTUM
# ============================================================================

print("\n[STEP 3] Dimensionality Reduction for Quantum...")
print("-"*70)

# Quantum circuits - using fewer but better qubits
# Use 5 qubits on the pre-selected discriminative features
n_qubits = 5  # 5 qubits for better feature representation

print(f"Selected features: {X_selected.shape[1]}")
print(f"Quantum qubits (reduced features): {n_qubits}")

pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X_selected)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

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
print(f"Training labels: {np.bincount(y_train)}")
print(f"Testing labels: {np.bincount(y_test)}")

# ============================================================================
# STEP 5: QUANTUM CIRCUIT SETUP
# ============================================================================

print("\n[STEP 5] Setting up Quantum Circuit...")
print("-"*70)

# Create feature map
feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=3,  # Increased to 3 repetitions for better feature encoding
    entanglement='full'  # Changed to full entanglement for richer feature space
)

print(f"Feature map: ZZFeatureMap")
print(f"  - Qubits: {n_qubits}")
print(f"  - Repetitions: 3")
print(f"  - Entanglement: full")

# Set backend (quantum simulator)
backend = AerSimulator()
print(f"Backend: Aer Simulator")

# ============================================================================
# STEP 6: CREATE QUANTUM KERNEL (SLOWEST PART)
# ============================================================================

print("\n[STEP 6] Creating Quantum Kernel...")
print("-"*70)
print("⏳ This may take a few minutes...")

start_time = time.time()

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print("Computing kernel matrix for training data...")
# Use more samples with stratified sampling for better representation
# Increased samples for better quantum model accuracy
max_samples = 600  # Increased to 600 for better training (300 per class)
n_class_0 = min(max_samples // 2, np.sum(y_train == 0))
n_class_1 = min(max_samples // 2, np.sum(y_train == 1))

# Get balanced indices with stratified sampling
idx_class_0 = np.where(y_train == 0)[0]
idx_class_1 = np.where(y_train == 1)[0]

# Randomly sample for diversity
np.random.seed(42)
idx_class_0 = np.random.choice(idx_class_0, n_class_0, replace=False)
idx_class_1 = np.random.choice(idx_class_1, n_class_1, replace=False)

balanced_indices = np.concatenate([idx_class_0, idx_class_1])
np.random.shuffle(balanced_indices)

X_train_quantum = X_train[balanced_indices]
y_train_quantum = y_train[balanced_indices]
print(f"Using {len(balanced_indices)} balanced samples for quantum training")
print(f"  Class 0: {np.sum(y_train_quantum == 0)}")
print(f"  Class 1: {np.sum(y_train_quantum == 1)}")

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

# Use sklearn SVC with precomputed quantum kernel
# Optimize hyperparameters for better classification
qsvc = SVC(kernel='precomputed', C=1000.0, class_weight='balanced', gamma='scale')
print("Fitting QSVC with quantum kernel...")
qsvc.fit(kernel_matrix, y_train_quantum)

training_time = time.time() - start_time
print(f"✓ QSVC training completed in {training_time:.2f} seconds")
print(f"Support vectors: {qsvc.n_support_}")

# ============================================================================
# STEP 8: EVALUATE QUANTUM MODEL
# ============================================================================

print("\n[STEP 8] Evaluating Quantum Model...")
print("-"*70)

# Predictions - compute kernel for test set
print("Computing kernel matrix for test data...")
# Get balanced test samples - increase for better evaluation
max_test = 200  # Increased to 200 for better evaluation (100 per class)
n_test_0 = min(max_test // 2, np.sum(y_test == 0))
n_test_1 = min(max_test // 2, np.sum(y_test == 1))

idx_test_0 = np.where(y_test == 0)[0][:n_test_0]
idx_test_1 = np.where(y_test == 1)[0][:n_test_1]
test_indices = np.concatenate([idx_test_0, idx_test_1])
np.random.shuffle(test_indices)

X_test_quantum = X_test[test_indices]
y_test_quantum = y_test[test_indices]
print(f"Using {len(test_indices)} balanced test samples")
print(f"  Class 0: {np.sum(y_test_quantum == 0)}")
print(f"  Class 1: {np.sum(y_test_quantum == 1)}")

kernel_test = quantum_kernel.evaluate(X_test_quantum, X_train_quantum)
y_pred_quantum = qsvc.predict(kernel_test)

print(f"\nPredictions distribution:")
print(f"  Predicted class 0: {np.sum(y_pred_quantum == 0)}")
print(f"  Predicted class 1: {np.sum(y_pred_quantum == 1)}")

# Metrics
quantum_accuracy = accuracy_score(y_test_quantum, y_pred_quantum)
quantum_precision = precision_score(y_test_quantum, y_pred_quantum, average='binary', zero_division=0)
quantum_recall = recall_score(y_test_quantum, y_pred_quantum, average='binary', zero_division=0)
quantum_f1 = f1_score(y_test_quantum, y_pred_quantum, average='binary', zero_division=0)

print(f"\n{'Metric':<20} {'Score':<15}")
print("-"*35)
print(f"{'Accuracy':<20} {quantum_accuracy:.4f}")
print(f"{'Precision':<20} {quantum_precision:.4f}")
print(f"{'Recall':<20} {quantum_recall:.4f}")
print(f"{'F1-Score':<20} {quantum_f1:.4f}")

# Confusion Matrix
cm_quantum = confusion_matrix(y_test_quantum, y_pred_quantum)
print(f"\nConfusion Matrix:\n{cm_quantum}")

print("\nClassification Report:")
print(classification_report(y_test_quantum, y_pred_quantum))

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n[STEP 9] Saving Results...")
print("-"*70)

quantum_results = {
    'model_type': 'QSVC',
    'accuracy': float(quantum_accuracy),
    'precision': float(quantum_precision),
    'recall': float(quantum_recall),
    'f1_score': float(quantum_f1),
    'kernel_computation_time': float(kernel_time),
    'training_time': float(training_time),
    'total_time': float(kernel_time + training_time),
    'qubits': int(n_qubits),
    'feature_map_reps': 3,
    'explained_variance': float(pca.explained_variance_ratio_.sum()),
    'training_samples': int(len(y_train_quantum)),
    'test_samples': int(len(y_test_quantum)),
    'C_parameter': 100.0,
    'class_weight': 'balanced'
}

with open('quantum_results.json', 'w') as f:
    json.dump(quantum_results, f, indent=4)

print("✓ Results saved to quantum_results.json")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================

print("\n[STEP 10] Creating Visualizations...")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_quantum, display_labels=['Class 0', 'Class 1'])
cm_display.plot(ax=axes[0, 0], cmap='Greens')
axes[0, 0].set_title('QSVC - Confusion Matrix', fontsize=12, fontweight='bold')

# Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [quantum_accuracy, quantum_precision, quantum_recall, quantum_f1]
axes[0, 1].bar(metrics_names, metrics_values, color='darkgreen', edgecolor='black', alpha=0.7)
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
    axes[1, 0].text(i, v + 0.3, f'{v:.1f}s', ha='center', fontweight='bold')

# PCA Explained Variance
axes[1, 1].plot(np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2, markersize=8)
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
📊 QSVC Results:
   Accuracy:  {quantum_accuracy:.4f}
   Precision: {quantum_precision:.4f}
   Recall:    {quantum_recall:.4f}
   F1-Score:  {quantum_f1:.4f}
   
⏱️  Quantum Kernel Time: {kernel_time:.2f} seconds
⏱️  QSVC Training Time: {training_time:.2f} seconds
⏱️  Total Time: {kernel_time + training_time:.2f} seconds
   
🔬 Model Details:
   Qubits (Features): {n_qubits}
   Feature Map Reps: 3
   Explained Variance: {pca.explained_variance_ratio_.sum():.4f}

📁 Output Files:
   ✓ quantum_results.json
   ✓ quantum_qsvc_results.png
""")

print("✓ Quantum model training complete!")