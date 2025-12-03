"""
Generate Additional Visualizations for Research Report
Includes: ROC curves, PR curves, feature analysis, error analysis, class distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*80)
print("GENERATING ADDITIONAL VISUALIZATIONS FOR RESEARCH REPORT")
print("="*80)

# ============================================================================
# LOAD RESULTS AND DATA
# ============================================================================

print("\n[LOADING] Results and predictions...")
print("-"*80)

# Load results
try:
    with open('classical_results.json', 'r') as f:
        classical_results = json.load(f)
    print("✓ Classical results loaded")
except FileNotFoundError:
    print("⚠ classical_results.json not found - some plots will be skipped")
    classical_results = None

try:
    with open('quantum_results.json', 'r') as f:
        quantum_results = json.load(f)
    print("✓ Quantum results loaded")
except FileNotFoundError:
    print("⚠ quantum_results.json not found - some plots will be skipped")
    quantum_results = None

# Load predictions if available
try:
    classical_preds = np.load('classical_predictions.npz')
    print("✓ Classical predictions loaded")
except:
    print("⚠ Classical predictions not found - will skip ROC/PR curves")
    classical_preds = None

try:
    quantum_preds = np.load('quantum_predictions.npz')
    print("✓ Quantum predictions loaded")
except:
    print("⚠ Quantum predictions not found - will skip ROC/PR curves")
    quantum_preds = None

# Load training history if available
try:
    with open('lstm_training_history.json', 'r') as f:
        training_history = json.load(f)
    print("✓ Training history loaded")
except:
    print("⚠ Training history not found - will skip loss curves")
    training_history = None

# ============================================================================
# PLOT 1: CLASS DISTRIBUTION
# ============================================================================

print("\n[PLOT 1] Generating Class Distribution...")
print("-"*80)

from load_climate_data import load_and_prepare_climate_data
X, y = load_and_prepare_climate_data()

if X is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    class_counts = np.bincount(y)
    class_labels = ['Normal Weather\n(DBZ ≤ 30)', 'Extreme Weather\n(DBZ > 30)']
    colors = ['#3498db', '#e74c3c']
    
    ax = axes[0]
    bars = ax.bar(class_labels, class_counts, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(class_counts, labels=class_labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90, 
                                        explode=(0, 0.1), shadow=True)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    ax.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    plt.close()
else:
    print("⚠ Could not load data - skipping class distribution plot")

# ============================================================================
# PLOT 2: FEATURE ANALYSIS (PCA Variance)
# ============================================================================

print("\n[PLOT 2] Generating PCA Feature Analysis...")
print("-"*80)

if X is not None:
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Variance explained by each component
    ax = axes[0]
    components = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)
    ax.bar(components[:20], pca_full.explained_variance_ratio_[:20], 
           color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Variance Explained by Each Component (Top 20)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Mark the 5 components used in quantum model
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Quantum Model Uses 5 Components')
    ax.legend(fontsize=10)
    
    # Cumulative variance
    ax = axes[1]
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(components, cumulative_variance, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1.5, label='95% Variance')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='5 Components (Quantum)')
    ax.fill_between(components[:5], 0, cumulative_variance[:5], alpha=0.3, color='red')
    
    # Add annotation for 5 components
    variance_5 = cumulative_variance[4]
    ax.annotate(f'5 Components:\n{variance_5:.2%} Variance',
                xy=(5, variance_5), xytext=(10, variance_5-0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red')
    
    ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('pca_feature_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pca_feature_analysis.png")
    print(f"  - First 5 components explain {variance_5:.2%} of variance")
    plt.close()
else:
    print("⚠ Could not load data - skipping PCA analysis")

# ============================================================================
# PLOT 3: ROC CURVES
# ============================================================================

print("\n[PLOT 3] Generating ROC Curves...")
print("-"*80)

if classical_preds is not None and quantum_preds is not None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Classical ROC
    y_true_classical = classical_preds['y_true']
    y_scores_classical = classical_preds['y_scores']
    fpr_classical, tpr_classical, _ = roc_curve(y_true_classical, y_scores_classical)
    roc_auc_classical = auc(fpr_classical, tpr_classical)
    
    ax.plot(fpr_classical, tpr_classical, color='#3498db', lw=3,
            label=f'Classical LSTM (AUC = {roc_auc_classical:.4f})')
    
    # Quantum ROC
    y_true_quantum = quantum_preds['y_true']
    y_scores_quantum = quantum_preds['y_scores']
    fpr_quantum, tpr_quantum, _ = roc_curve(y_true_quantum, y_scores_quantum)
    roc_auc_quantum = auc(fpr_quantum, tpr_quantum)
    
    ax.plot(fpr_quantum, tpr_quantum, color='#e74c3c', lw=3,
            label=f'Quantum QSVC (AUC = {roc_auc_quantum:.4f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves: Classical vs Quantum Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curves.png")
    print(f"  - Classical AUC: {roc_auc_classical:.4f}")
    print(f"  - Quantum AUC: {roc_auc_quantum:.4f}")
    plt.close()
else:
    print("⚠ Predictions not available - skipping ROC curves")
    print("  To generate ROC curves, re-run models with prediction saving enabled")

# ============================================================================
# PLOT 4: PRECISION-RECALL CURVES
# ============================================================================

print("\n[PLOT 4] Generating Precision-Recall Curves...")
print("-"*80)

if classical_preds is not None and quantum_preds is not None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Classical PR
    precision_classical, recall_classical, _ = precision_recall_curve(y_true_classical, y_scores_classical)
    avg_precision_classical = average_precision_score(y_true_classical, y_scores_classical)
    
    ax.plot(recall_classical, precision_classical, color='#3498db', lw=3,
            label=f'Classical LSTM (AP = {avg_precision_classical:.4f})')
    
    # Quantum PR
    precision_quantum, recall_quantum, _ = precision_recall_curve(y_true_quantum, y_scores_quantum)
    avg_precision_quantum = average_precision_score(y_true_quantum, y_scores_quantum)
    
    ax.plot(recall_quantum, precision_quantum, color='#e74c3c', lw=3,
            label=f'Quantum QSVC (AP = {avg_precision_quantum:.4f})')
    
    # Baseline (prevalence of positive class)
    baseline = np.sum(y_true_classical) / len(y_true_classical)
    ax.axhline(y=baseline, color='k', linestyle='--', lw=2, 
               label=f'Baseline (Prevalence = {baseline:.2%})')
    
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curves: Classical vs Quantum Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: precision_recall_curves.png")
    print(f"  - Classical AP: {avg_precision_classical:.4f}")
    print(f"  - Quantum AP: {avg_precision_quantum:.4f}")
    plt.close()
else:
    print("⚠ Predictions not available - skipping PR curves")

# ============================================================================
# PLOT 5: TRAINING HISTORY (LSTM Loss Curves)
# ============================================================================

print("\n[PLOT 5] Generating Training History...")
print("-"*80)

if training_history is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(training_history['loss']) + 1)
    
    # Loss curve
    ax = axes[0]
    ax.plot(epochs, training_history['loss'], 'o-', color='#3498db', 
            linewidth=2, markersize=5, label='Training Loss')
    ax.plot(epochs, training_history['val_loss'], 's-', color='#e74c3c', 
            linewidth=2, markersize=5, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title('LSTM Training: Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Mark early stopping point if available
    if 'stopped_epoch' in training_history:
        ax.axvline(x=training_history['stopped_epoch'], color='green', 
                   linestyle='--', linewidth=2, label=f"Early Stop (Epoch {training_history['stopped_epoch']})")
    
    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, training_history['accuracy'], 'o-', color='#3498db', 
            linewidth=2, markersize=5, label='Training Accuracy')
    ax.plot(epochs, training_history['val_accuracy'], 's-', color='#e74c3c', 
            linewidth=2, markersize=5, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('LSTM Training: Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lstm_training_history.png")
    plt.close()
else:
    print("⚠ Training history not available - skipping loss curves")
    print("  To generate loss curves, re-run classical model with history saving enabled")

# ============================================================================
# PLOT 6: ERROR ANALYSIS
# ============================================================================

print("\n[PLOT 6] Generating Error Analysis...")
print("-"*80)

if classical_preds is not None and quantum_preds is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Classical confusion matrix breakdown
    y_pred_classical = classical_preds['y_pred']
    cm_classical = classical_results['confusion_matrix'] if classical_results else None
    
    if cm_classical:
        tn_c, fp_c, fn_c, tp_c = cm_classical[0][0], cm_classical[0][1], cm_classical[1][0], cm_classical[1][1]
        
        ax = axes[0, 0]
        error_types = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
        counts = [tn_c, fp_c, fn_c, tp_c]
        colors_err = ['#2ecc71', '#e74c3c', '#e67e22', '#2ecc71']
        
        bars = ax.bar(error_types, counts, color=colors_err, alpha=0.7, edgecolor='black', width=0.6)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Classical LSTM: Prediction Breakdown', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Quantum confusion matrix breakdown
    if quantum_preds is not None:
        from sklearn.metrics import confusion_matrix as cm_func
        cm_quantum = cm_func(quantum_preds['y_true'], quantum_preds['y_pred'])
        tn_q, fp_q, fn_q, tp_q = cm_quantum[0][0], cm_quantum[0][1], cm_quantum[1][0], cm_quantum[1][1]
        
        ax = axes[0, 1]
        counts_q = [tn_q, fp_q, fn_q, tp_q]
        bars = ax.bar(error_types, counts_q, color=colors_err, alpha=0.7, edgecolor='black', width=0.6)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Quantum QSVC: Prediction Breakdown', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts_q):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Error rate comparison
    if cm_classical and quantum_preds is not None:
        ax = axes[1, 0]
        
        total_classical = sum(counts)
        total_quantum = sum(counts_q)
        
        fp_rate_c = fp_c / (fp_c + tn_c) * 100
        fn_rate_c = fn_c / (fn_c + tp_c) * 100
        fp_rate_q = fp_q / (fp_q + tn_q) * 100
        fn_rate_q = fn_q / (fn_q + tp_q) * 100
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [fp_rate_c, fn_rate_c], width, 
                       label='Classical LSTM', color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, [fp_rate_q, fn_rate_q], width, 
                       label='Quantum QSVC', color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Error Rate Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['False Positive\nRate', 'False Negative\nRate'])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Model comparison summary
    ax = axes[1, 1]
    ax.axis('off')
    
    if classical_results and quantum_results:
        summary_text = f"""
MODEL PERFORMANCE SUMMARY

Classical LSTM:
  • Accuracy:  {classical_results['accuracy']:.4f}
  • Precision: {classical_results['precision']:.4f}
  • Recall:    {classical_results['recall']:.4f}
  • F1-Score:  {classical_results['f1_score']:.4f}
  
  Error Analysis:
  • False Positives: {fp_c:,} ({fp_rate_c:.2f}%)
  • False Negatives: {fn_c:,} ({fn_rate_c:.2f}%)

Quantum QSVC:
  • Accuracy:  {quantum_results['accuracy']:.4f}
  • Precision: {quantum_results['precision']:.4f}
  • Recall:    {quantum_results['recall']:.4f}
  • F1-Score:  {quantum_results['f1_score']:.4f}
  
  Error Analysis:
  • False Positives: {fp_q:,} ({fp_rate_q:.2f}%)
  • False Negatives: {fn_q:,} ({fn_rate_q:.2f}%)

Key Insights:
  • Both models excel at identifying normal weather
  • Extreme weather detection varies by model
  • Class imbalance affects precision/recall trade-off
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: error_analysis.png")
    plt.close()
else:
    print("⚠ Predictions not available - skipping error analysis")

# ============================================================================
# PLOT 7: QUANTUM CIRCUIT VISUALIZATION
# ============================================================================

print("\n[PLOT 7] Generating Quantum Circuit Diagram...")
print("-"*80)

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.visualization import circuit_drawer
    
    # Create the feature map used in quantum model
    feature_map = ZZFeatureMap(feature_dimension=5, reps=2, entanglement='full')
    
    # Draw circuit
    fig = circuit_drawer(feature_map, output='mpl', style='iqp', fold=20)
    fig.savefig('quantum_circuit_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: quantum_circuit_diagram.png")
    print("  - 5 qubits, 2 repetitions, full entanglement")
    plt.close()
except Exception as e:
    print(f"⚠ Could not generate circuit diagram: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*80)

generated_files = [
    "class_distribution.png",
    "pca_feature_analysis.png",
    "roc_curves.png (if predictions available)",
    "precision_recall_curves.png (if predictions available)",
    "lstm_training_history.png (if history available)",
    "error_analysis.png (if predictions available)",
    "quantum_circuit_diagram.png"
]

print("\nGenerated files:")
for f in generated_files:
    print(f"  • {f}")

print("\n" + "="*80)
print("NOTE: Some plots require re-running models with additional data saving.")
print("Run the updated model scripts to generate all visualizations.")
print("="*80)
