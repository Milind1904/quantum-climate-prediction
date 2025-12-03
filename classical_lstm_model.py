import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import time
import json

print("="*70)
print("CLASSICAL MODEL: LSTM on NVIDIA GPU")
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
# STEP 2: VERIFY GPU AVAILABILITY
# ============================================================================

print("\n[STEP 2] GPU Configuration...")
print("-"*70)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU Details: {gpu_devices}")
    print("✓ GPU will be used for training")
else:
    print("⚠️  No GPU detected. Training will use CPU.")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n[STEP 3] Data Preprocessing...")
print("-"*70)

# Check for missing values
if np.any(np.isnan(X)):
    print("⚠️  Missing values detected. Filling with mean...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

# Normalize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features shape after scaling: {X_scaled.shape}")
print(f"Feature statistics - Mean: {X_scaled.mean(axis=0)[:3]}, Std: {X_scaled.std(axis=0)[:3]}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================

print("\n[STEP 4] Train-Test Split...")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print(f"Training labels: {np.bincount(y_train)}")
print(f"Testing labels: {np.bincount(y_test)}")

# ============================================================================
# STEP 5: RESHAPE FOR LSTM
# ============================================================================

print("\n[STEP 5] Reshaping Data for LSTM...")
print("-"*70)

# Reshape to (samples, timesteps, features)
# Using 1 timestep
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"X_train_lstm shape: {X_train_lstm.shape}")
print(f"X_test_lstm shape: {X_test_lstm.shape}")

# ============================================================================
# STEP 6: BUILD LSTM MODEL
# ============================================================================

print("\n[STEP 6] Building LSTM Model...")
print("-"*70)

model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
print(model.summary())

# ============================================================================
# STEP 7: TRAIN MODEL ON GPU
# ============================================================================

print("\n[STEP 7] Training LSTM Model...")
print("-"*70)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()

with tf.device('/GPU:0'):
    history = model.fit(
        X_train_lstm, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================

print("\n[STEP 8] Evaluating LSTM Model...")
print("-"*70)

# Predictions
y_scores_lstm = model.predict(X_test_lstm, verbose=0).flatten()  # Save raw scores for ROC/PR
y_pred_lstm = (y_scores_lstm > 0.5).astype(int)

# Metrics
lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
lstm_precision = precision_score(y_test, y_pred_lstm, zero_division=0)
lstm_recall = recall_score(y_test, y_pred_lstm, zero_division=0)
lstm_f1 = f1_score(y_test, y_pred_lstm, zero_division=0)

print(f"\n{'Metric':<20} {'Score':<15}")
print("-"*35)
print(f"{'Accuracy':<20} {lstm_accuracy:.4f}")
print(f"{'Precision':<20} {lstm_precision:.4f}")
print(f"{'Recall':<20} {lstm_recall:.4f}")
print(f"{'F1-Score':<20} {lstm_f1:.4f}")

# Confusion Matrix
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
print(f"\nConfusion Matrix:\n{cm_lstm}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lstm))

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n[STEP 9] Saving Results...")
print("-"*70)

classical_results = {
    'model_type': 'LSTM',
    'accuracy': float(lstm_accuracy),
    'precision': float(lstm_precision),
    'recall': float(lstm_recall),
    'f1_score': float(lstm_f1),
    'training_time_seconds': float(training_time),
    'parameters': int(model.count_params()),
    'epochs_trained': len(history.history['loss']),
    'confusion_matrix': cm_lstm.tolist()
}

with open('classical_results.json', 'w') as f:
    json.dump(classical_results, f, indent=4)

print("✓ Results saved to classical_results.json")

# Save training history for visualization
training_hist = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'stopped_epoch': len(history.history['loss'])
}

with open('lstm_training_history.json', 'w') as f:
    json.dump(training_hist, f, indent=4)
print("✓ Training history saved to lstm_training_history.json")

# Save predictions for ROC/PR curves
np.savez('classical_predictions.npz', 
         y_true=y_test, 
         y_pred=y_pred_lstm,
         y_scores=y_scores_lstm)
print("✓ Predictions saved to classical_predictions.npz")

# Save model
model.save('lstm_model.h5')
print("✓ Model saved to lstm_model.h5")

# Save scaler for future use
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to scaler.pkl")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================

print("\n[STEP 10] Creating Visualizations...")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training History - Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('LSTM - Loss Over Epochs', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training History - Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('LSTM - Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=['Class 0', 'Class 1'])
cm_display.plot(ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('LSTM - Confusion Matrix', fontsize=12, fontweight='bold')

# Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1]
axes[1, 1].bar(metrics_names, metrics_values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('LSTM - Performance Metrics', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim([0, 1.0])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('classical_lstm_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: classical_lstm_results.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("CLASSICAL MODEL TRAINING COMPLETE")
print("="*70)
print(f"""
📊 LSTM Results:
   Accuracy:  {lstm_accuracy:.4f}
   Precision: {lstm_precision:.4f}
   Recall:    {lstm_recall:.4f}
   F1-Score:  {lstm_f1:.4f}
   
⏱️  Training Time: {training_time:.2f} seconds
📦 Model Parameters: {model.count_params():,}

📁 Output Files:
   ✓ classical_results.json
   ✓ lstm_model.h5
   ✓ scaler.pkl
   ✓ classical_lstm_results.png
""")

# Return results for potential use in comparison
print("✓ Classical model training complete!")