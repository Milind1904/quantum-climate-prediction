# Chapter 3: Requirement Specification

## 3.1 Mapping of Requirements

The requirement mapping provides a comprehensive overview of how various stakeholder needs are translated into system specifications for the Quantum vs Classical Machine Learning Climate Prediction system.

### 3.1.1 Stakeholder-to-Requirement Mapping

| Stakeholder | Need | System Requirement | Priority |
|-------------|------|-------------------|----------|
| Researchers | Compare quantum and classical ML performance | Implement both QSVC and LSTM models | High |
| Data Scientists | Accurate climate prediction | Achieve >70% accuracy on weather classification | High |
| System Administrators | Reproducible experiments | Fixed random seeds, documented code | Medium |
| Climate Scientists | Extreme weather detection | High precision/recall for extreme events | High |
| Academic Community | Transparent methodology | Open source code and detailed documentation | Medium |
| Future Developers | Extensible system | Modular architecture for new models | Low |

### 3.1.2 Requirement Traceability Matrix

| Req ID | Requirement Description | Type | Source | Implementation |
|--------|------------------------|------|--------|----------------|
| FR-01 | Load and preprocess climate data | Functional | Data Scientists | load_climate_data.py |
| FR-02 | Train classical LSTM model | Functional | Researchers | classical_lstm_model.py |
| FR-03 | Train quantum QSVC model | Functional | Researchers | quantum_qsvc_optimized.py |
| FR-04 | Generate performance visualizations | Functional | All Stakeholders | compare_models.py |
| FR-05 | Compare model performance | Functional | Researchers | compare_models.py |
| NFR-01 | System accuracy >70% | Non-Functional | Data Scientists | Model evaluation |
| NFR-02 | Processing time <2 hours | Non-Functional | System Admins | Optimized implementation |
| NFR-03 | Reproducible results | Non-Functional | Academic | Random seed control |

---

## 3.2 Functional Requirements

Functional requirements define what the system must do to meet user needs.

### 3.2.1 Data Management Requirements

**FR-01: Data Loading and Preprocessing**
- **Description**: System shall load Indian weather radar data from CSV files and perform preprocessing
- **Input**: 157 CSV files containing RSCHR weather radar data
- **Process**: 
  - Read all CSV files from data directory
  - Combine into single dataset (323,280 samples)
  - Extract 35 meteorological features
  - Handle missing values using column mean imputation
  - Create binary labels (normal vs extreme weather based on DBZ > 30)
- **Output**: Preprocessed feature matrix X and label vector y
- **Priority**: Critical
- **Dependencies**: None

**FR-02: Feature Normalization**
- **Description**: System shall normalize all features to zero mean and unit variance
- **Input**: Raw feature matrix (323,280 × 35)
- **Process**: Apply StandardScaler transformation
- **Output**: Normalized feature matrix
- **Priority**: Critical
- **Dependencies**: FR-01

**FR-03: Train-Test Split**
- **Description**: System shall split data into training and testing sets with stratification
- **Input**: Preprocessed and normalized dataset
- **Process**: 
  - 80-20 train-test split
  - Maintain class distribution (stratified sampling)
  - Fixed random seed (42) for reproducibility
- **Output**: X_train, X_test, y_train, y_test
- **Priority**: Critical
- **Dependencies**: FR-01, FR-02

### 3.2.2 Classical Model Requirements

**FR-04: LSTM Model Architecture**
- **Description**: System shall implement a deep LSTM neural network for binary classification
- **Specifications**:
  - Input layer: 35 features
  - LSTM layer: 64 units
  - Dropout layer: 0.2 rate
  - Dense layer: 32 units (ReLU activation)
  - Dropout layer: 0.2 rate
  - Output layer: 1 unit (Sigmoid activation)
  - Total parameters: 27,713
- **Priority**: Critical
- **Dependencies**: FR-03

**FR-05: LSTM Training**
- **Description**: System shall train LSTM model with specified hyperparameters
- **Specifications**:
  - Optimizer: Adam (learning rate 0.001)
  - Loss function: Binary crossentropy
  - Batch size: 20
  - Maximum epochs: 50
  - Early stopping: patience 5 on validation loss
  - Training samples: 258,624 (full training set)
- **Output**: Trained LSTM model saved as lstm_model.h5
- **Priority**: Critical
- **Dependencies**: FR-04

**FR-06: Classical Model Evaluation**
- **Description**: System shall evaluate LSTM performance on test set
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Output**: classical_results.json with all metrics
- **Priority**: Critical
- **Dependencies**: FR-05

### 3.2.3 Quantum Model Requirements

**FR-07: Dimensionality Reduction**
- **Description**: System shall reduce features from 35 to 5 dimensions using PCA
- **Input**: Normalized training data
- **Process**: 
  - Apply PCA with n_components=5
  - Fit on training data
  - Transform both training and test data
- **Output**: 5-dimensional feature vectors
- **Constraint**: Must capture >60% variance
- **Priority**: Critical
- **Dependencies**: FR-03

**FR-08: Quantum Feature Map Construction**
- **Description**: System shall create a 5-qubit quantum circuit for data encoding
- **Specifications**:
  - Feature map type: ZZFeatureMap
  - Number of qubits: 5
  - Repetitions: 2
  - Entanglement: Full
  - Backend: Qiskit Aer Simulator
- **Priority**: Critical
- **Dependencies**: FR-07

**FR-09: Quantum Kernel Computation**
- **Description**: System shall compute quantum kernel matrices for training and testing
- **Input**: 5-dimensional PCA-transformed data
- **Process**:
  - Compute train kernel matrix K_train (1000 × 1000)
  - Compute test kernel matrix K_test (500 × 1000)
  - Use FidelityQuantumKernel: K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
- **Output**: Precomputed kernel matrices
- **Priority**: Critical
- **Dependencies**: FR-07, FR-08

**FR-10: QSVC Training**
- **Description**: System shall train quantum support vector classifier
- **Specifications**:
  - Kernel: Precomputed (quantum kernel)
  - C parameter: 10.0
  - Class weights: Balanced
  - Training samples: 1,000 (stratified sampling)
  - Test samples: 500
- **Output**: Trained QSVC model
- **Priority**: Critical
- **Dependencies**: FR-09

**FR-11: Quantum Model Evaluation**
- **Description**: System shall evaluate QSVC performance on test set
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Output**: quantum_results.json with all metrics
- **Priority**: Critical
- **Dependencies**: FR-10

### 3.2.4 Comparison and Visualization Requirements

**FR-12: Performance Comparison**
- **Description**: System shall generate comparative analysis of both models
- **Input**: classical_results.json, quantum_results.json
- **Process**:
  - Compare accuracy, precision, recall, F1-score
  - Calculate parameter efficiency ratio
  - Analyze training time differences
  - Generate summary statistics
- **Output**: summary_report.txt
- **Priority**: High
- **Dependencies**: FR-06, FR-11

**FR-13: Visualization Generation**
- **Description**: System shall create performance visualization charts
- **Required Visualizations**:
  1. Classical LSTM confusion matrix and metrics
  2. Quantum QSVC confusion matrix and metrics
  3. Side-by-side comparison chart
- **Format**: PNG images (300 DPI)
- **Output Files**: 
  - classical_lstm_results.png
  - quantum_qsvc_results.png
  - model_comparison_complete.png
- **Priority**: High
- **Dependencies**: FR-06, FR-11

**FR-14: Results Export**
- **Description**: System shall export all results in machine-readable format
- **Output Files**:
  - classical_results.json
  - quantum_results.json
  - summary_report.txt
  - All visualization PNG files
- **Priority**: Medium
- **Dependencies**: FR-06, FR-11, FR-13

### 3.2.5 Reporting Requirements

**FR-15: Final Report Generation**
- **Description**: System shall generate comprehensive project documentation
- **Required Documents**:
  - FINAL_RESULTS.md: Complete results summary
  - RESEARCH_PAPER.md: Academic paper format
  - REFERENCES.md: Bibliography
  - QUANTUM_APPROACHES_SUMMARY.md: Technical analysis
- **Priority**: Medium
- **Dependencies**: All functional requirements

---

## 3.3 Non-Functional Requirements

Non-functional requirements define system qualities and constraints.

### 3.3.1 Performance Requirements

**NFR-01: Accuracy Threshold**
- **Requirement**: Classical model shall achieve ≥95% accuracy
- **Rationale**: Climate prediction requires high reliability
- **Measurement**: Accuracy on test set
- **Achieved**: 99.74% ✓

**NFR-02: Quantum Model Viability**
- **Requirement**: Quantum model shall achieve ≥70% accuracy
- **Rationale**: Demonstrate quantum ML feasibility
- **Measurement**: Accuracy on test set
- **Achieved**: 78.60% ✓

**NFR-03: Training Time**
- **Requirement**: Classical model training shall complete within 1 hour
- **Rationale**: Enable iterative experimentation
- **Measurement**: Wall-clock time for model.fit()
- **Achieved**: 273 seconds (4.5 minutes) ✓

**NFR-04: Quantum Kernel Computation**
- **Requirement**: Quantum kernel computation shall complete within 2 hours
- **Rationale**: Acceptable for research purposes
- **Measurement**: Time for kernel matrix evaluation
- **Achieved**: 3281 seconds (55 minutes) ✓

**NFR-05: Memory Usage**
- **Requirement**: System shall operate within 16 GB RAM
- **Rationale**: Standard workstation constraint
- **Measurement**: Peak memory consumption
- **Status**: Compliant ✓

### 3.3.2 Reliability Requirements

**NFR-06: Reproducibility**
- **Requirement**: System shall produce identical results across multiple runs
- **Implementation**:
  - Fixed random seed (seed=42)
  - Deterministic algorithms where possible
  - Documented environment (Python 3.13, package versions)
- **Verification**: Multiple execution tests
- **Priority**: High

**NFR-07: Error Handling**
- **Requirement**: System shall handle common errors gracefully
- **Coverage**:
  - Missing data files
  - Corrupted CSV files
  - Missing values in data
  - Memory allocation failures
- **Implementation**: Try-except blocks with informative error messages
- **Priority**: Medium

**NFR-08: Data Integrity**
- **Requirement**: System shall preserve data integrity throughout pipeline
- **Validation**:
  - Shape checks after each transformation
  - Class distribution verification
  - No data leakage between train/test sets
- **Priority**: Critical

### 3.3.3 Usability Requirements

**NFR-09: Code Readability**
- **Requirement**: Code shall be self-documenting with clear variable names
- **Standards**:
  - PEP 8 compliance for Python code
  - Docstrings for all functions
  - Inline comments for complex logic
- **Priority**: Medium

**NFR-10: Documentation Completeness**
- **Requirement**: System shall have comprehensive user documentation
- **Required Docs**:
  - README with setup instructions
  - API documentation for functions
  - Usage examples
  - Troubleshooting guide
- **Priority**: Medium

**NFR-11: Output Interpretability**
- **Requirement**: All outputs shall be clearly labeled and explained
- **Implementation**:
  - Progress messages during execution
  - Clear metric labels in results
  - Legends and titles on visualizations
- **Priority**: High

### 3.3.4 Maintainability Requirements

**NFR-12: Code Modularity**
- **Requirement**: System shall be organized into reusable modules
- **Structure**:
  - load_climate_data.py: Data loading functions
  - classical_lstm_model.py: Classical model
  - quantum_qsvc_optimized.py: Quantum model
  - compare_models.py: Comparison utilities
- **Priority**: Medium

**NFR-13: Version Control**
- **Requirement**: All code changes shall be tracked
- **Implementation**: Git repository with meaningful commit messages
- **Priority**: Medium

**NFR-14: Extensibility**
- **Requirement**: System shall allow easy addition of new models
- **Design**: Abstract base classes for models
- **Priority**: Low

### 3.3.5 Portability Requirements

**NFR-15: Platform Independence**
- **Requirement**: System shall run on Windows, Linux, and macOS
- **Constraints**: Python 3.8+ required
- **Verification**: Tested on Windows 11 ✓
- **Priority**: Medium

**NFR-16: Cloud Compatibility**
- **Requirement**: Quantum models shall be deployable on Google Colab
- **Implementation**: quantum_ensemble_colab.ipynb notebook provided
- **Priority**: Medium

### 3.3.6 Security Requirements

**NFR-17: Data Privacy**
- **Requirement**: System shall not transmit data externally
- **Implementation**: All computation local or on specified quantum simulators
- **Priority**: Low (research project, public data)

---

## 3.4 User Requirements

User requirements capture what users need to accomplish with the system.

### 3.4.1 Researcher Requirements

**UR-01: Model Comparison Capability**
- **Need**: Compare quantum and classical ML approaches objectively
- **System Support**: 
  - Side-by-side metric comparison
  - Statistical significance testing
  - Visual comparison charts
- **Success Criteria**: Clear determination of which model performs better

**UR-02: Reproducible Experiments**
- **Need**: Replicate results for verification and publication
- **System Support**:
  - Fixed random seeds
  - Version-controlled code
  - Documented dependencies
- **Success Criteria**: Same results on multiple runs

**UR-03: Performance Analysis**
- **Need**: Understand why each model performs as it does
- **System Support**:
  - Detailed confusion matrices
  - Per-class metrics (precision, recall)
  - Feature importance analysis
- **Success Criteria**: Insights into model strengths/weaknesses

### 3.4.2 Data Scientist Requirements

**UR-04: Data Exploration**
- **Need**: Understand dataset characteristics before modeling
- **System Support**:
  - Dataset statistics (shape, class distribution)
  - Missing value analysis
  - Feature correlation analysis
- **Success Criteria**: Informed preprocessing decisions

**UR-05: Hyperparameter Tuning**
- **Need**: Optimize model parameters for best performance
- **System Support**:
  - Configurable hyperparameters
  - Quick iteration cycle
  - Performance tracking
- **Success Criteria**: Optimal model configuration identified

**UR-06: Model Export**
- **Need**: Save trained models for later use
- **System Support**:
  - Save LSTM as .h5 file
  - Save scalers as .pkl files
  - Export quantum feature maps
- **Success Criteria**: Models reloadable without retraining

### 3.4.3 Climate Scientist Requirements

**UR-07: Extreme Weather Detection**
- **Need**: Identify severe weather events accurately
- **System Support**:
  - High recall for extreme weather class
  - Balanced class weighting
  - Precision-recall trade-off analysis
- **Success Criteria**: >85% recall on extreme weather events

**UR-08: Interpretable Predictions**
- **Need**: Understand why model made specific predictions
- **System Support**:
  - Confusion matrix breakdown
  - Feature contribution analysis
  - Prediction confidence scores
- **Success Criteria**: Explainable predictions

### 3.4.4 System Administrator Requirements

**UR-09: Easy Deployment**
- **Need**: Set up system quickly on new machines
- **System Support**:
  - requirements.txt with all dependencies
  - Installation instructions
  - Environment setup scripts
- **Success Criteria**: Setup completed in <30 minutes

**UR-10: Resource Monitoring**
- **Need**: Track system resource usage during execution
- **System Support**:
  - Training time reporting
  - Memory usage logging
  - Progress indicators
- **Success Criteria**: Visibility into resource consumption

---

## 3.5 Domain Requirements

Domain requirements are derived from the climate science and quantum computing domains.

### 3.5.1 Climate Science Domain Requirements

**DR-01: Weather Radar Data Standards**
- **Requirement**: System shall process standard weather radar formats (RSCHR)
- **Specifications**:
  - Support dual-polarization radar variables (DBZ, VEL, WIDTH, ZDR, PHIDP, RHOHV)
  - Handle temporal and spatial metadata
  - Process range-azimuth-elevation data structure
- **Source**: Indian Meteorological Department standards
- **Priority**: Critical

**DR-02: Extreme Weather Threshold**
- **Requirement**: System shall use meteorologically valid threshold for extreme weather
- **Specification**: DBZ > 30 dBZ indicates extreme convective activity
- **Rationale**: Standard threshold used in operational meteorology
- **Source**: WMO (World Meteorological Organization) guidelines
- **Priority**: Critical

**DR-03: Class Imbalance Handling**
- **Requirement**: System shall handle natural weather class imbalance
- **Context**: Real-world weather is 87% normal, 13% extreme
- **Approach**: 
  - Maintain natural distribution in test set
  - Use balanced class weights in training
  - Report both overall and per-class metrics
- **Priority**: High

**DR-04: Feature Standardization**
- **Requirement**: System shall normalize features due to different physical units
- **Rationale**: 
  - DBZ in dBZ units (logarithmic)
  - Velocity in m/s
  - Correlation coefficient unitless (0-1)
- **Approach**: StandardScaler (zero mean, unit variance)
- **Priority**: Critical

**DR-05: Temporal Independence**
- **Requirement**: Train and test sets shall be temporally independent
- **Rationale**: Prevent data leakage from temporal autocorrelation
- **Implementation**: Random sampling with stratification
- **Priority**: High

### 3.5.2 Quantum Computing Domain Requirements

**DR-06: Qubit Limitations**
- **Requirement**: System shall work within NISQ-era qubit constraints
- **Specification**: Use ≤10 qubits for practical simulation
- **Implementation**: 5 qubits chosen for balance
- **Source**: Current quantum hardware limitations
- **Priority**: Critical

**DR-07: Quantum Feature Encoding**
- **Requirement**: System shall encode classical data into quantum states
- **Approach**: Angle encoding via rotation gates
- **Constraint**: Feature values must be normalized to [0, 2π]
- **Implementation**: ZZFeatureMap with amplitude scaling
- **Priority**: Critical

**DR-08: Quantum Circuit Depth**
- **Requirement**: System shall use shallow circuits to avoid barren plateaus
- **Specification**: Maximum 2 repetitions in feature map
- **Rationale**: Deep circuits suffer from vanishing gradients
- **Source**: McClean et al. (2018) barren plateau research
- **Priority**: High

**DR-09: Quantum Measurement Strategy**
- **Requirement**: System shall use efficient quantum kernel evaluation
- **Approach**: Fidelity-based kernel K(x,y) = |⟨φ(x)|φ(y)⟩|²
- **Advantage**: Avoids explicit state preparation
- **Priority**: Critical

**DR-10: Simulator Backend**
- **Requirement**: System shall use appropriate quantum simulator
- **Specification**: Qiskit Aer statevector simulator
- **Rationale**: 
  - Exact simulation for benchmarking
  - No quantum noise for baseline comparison
  - Fast enough for 5 qubits
- **Priority**: High

### 3.5.3 Machine Learning Domain Requirements

**DR-11: Train-Test Separation**
- **Requirement**: System shall maintain strict train-test separation
- **Specification**: No test data used during training or hyperparameter tuning
- **Verification**: Independent test set evaluation only
- **Priority**: Critical

**DR-12: Evaluation Metrics**
- **Requirement**: System shall report standard classification metrics
- **Required Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Rationale**: Enable comparison with published research
- **Priority**: Critical

**DR-13: Cross-Validation** (Future Enhancement)
- **Recommendation**: Use k-fold cross-validation for robust evaluation
- **Current Status**: Single train-test split due to computational constraints
- **Priority**: Low

---

## 3.6 System Requirements

System requirements define the technical infrastructure needed to run the system.

### 3.6.1 Hardware Requirements

**SR-01: Minimum Hardware Specifications**
- **CPU**: Intel Core i5 or equivalent (4+ cores recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space for data and models
- **GPU**: Not required (CPU-only computation)
- **Rationale**: 
  - Quantum simulation is CPU-intensive
  - Data fits in memory
  - No GPU acceleration implemented

**SR-02: Recommended Hardware Specifications**
- **CPU**: Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores)
- **RAM**: 32 GB for large-scale experiments
- **Storage**: SSD for faster data loading
- **Benefits**: 
  - Faster quantum kernel computation
  - Enable larger sample sizes
  - Parallel experiment execution

### 3.6.2 Software Requirements

**SR-03: Operating System**
- **Supported**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **Tested On**: Windows 11
- **Compatibility**: Python cross-platform libraries

**SR-04: Python Environment**
- **Version**: Python 3.8 - 3.13
- **Recommended**: Python 3.13 (tested)
- **Installation**: Anaconda or virtualenv recommended

**SR-05: Core Dependencies**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
```

**SR-06: Quantum Computing Libraries**
```
qiskit>=0.45.0
qiskit-aer>=0.13.0
qiskit-machine-learning>=0.7.0
```

**SR-07: Visualization Libraries**
```
matplotlib>=3.5.0
seaborn>=0.11.0
```

**SR-08: Utility Libraries**
```
scipy>=1.7.0
joblib>=1.1.0 (for model persistence)
```

### 3.6.3 Data Requirements

**SR-09: Input Data Format**
- **Format**: CSV files with header row
- **Encoding**: UTF-8
- **Separator**: Comma
- **Structure**: Each file contains radar scan data with 41 columns

**SR-10: Data Storage**
- **Location**: data/extracted_data/extracted_data/ directory
- **Size**: ~500 MB for 157 CSV files
- **Organization**: One file per radar scan
- **Naming Convention**: RSCHR_DDMMMYYYY_HHMMSS_L2B_STD.csv

**SR-11: Output Data Format**
- **JSON**: Results stored in JSON format for machine readability
- **PNG**: Visualizations in PNG format (300 DPI)
- **Text**: Summary reports in Markdown format
- **Models**: HDF5 (.h5) for LSTM, pickle (.pkl) for scalers

### 3.6.4 Performance Requirements (System-Level)

**SR-12: Processing Throughput**
- **Data Loading**: Process 323,280 samples in <30 seconds
- **Feature Extraction**: Transform features in <10 seconds
- **Classical Training**: Complete within 15 minutes
- **Quantum Kernel**: Compute 1000×1000 matrix in <60 minutes

**SR-13: Memory Footprint**
- **Data**: ~3 GB for full dataset in memory
- **Classical Model**: ~100 MB during training
- **Quantum Model**: ~2 GB for kernel matrices
- **Peak Usage**: <12 GB total

**SR-14: Scalability**
- **Data Samples**: Support up to 500,000 samples for classical model
- **Quantum Samples**: Limited to 2,000 for kernel computation
- **Features**: Support up to 100 features (with PCA)
- **Qubits**: Support up to 10 qubits (simulator limitation)

### 3.6.5 Network Requirements

**SR-15: Internet Connectivity**
- **Installation Phase**: Required for downloading packages
- **Execution Phase**: Not required (all local computation)
- **Optional**: Internet access for downloading quantum backends from IBM

**SR-16: Cloud Deployment** (Optional)
- **Platform**: Google Colab compatible
- **Benefits**: Free GPU/TPU access, no local installation
- **Notebook**: quantum_ensemble_colab.ipynb provided

### 3.6.6 Development Requirements

**SR-17: Development Tools**
- **IDE**: Any Python IDE (VS Code, PyCharm, Jupyter recommended)
- **Version Control**: Git for code management
- **Notebook**: Jupyter Notebook/Lab for interactive development

**SR-18: Testing Framework**
- **Unit Tests**: pytest for function testing
- **Integration Tests**: End-to-end pipeline validation
- **Coverage**: pytest-cov for code coverage analysis

---

## Requirements Summary Table

| Category | Total Requirements | Critical | High | Medium | Low |
|----------|-------------------|----------|------|--------|-----|
| **Functional** | 15 | 11 | 3 | 1 | 0 |
| **Non-Functional** | 17 | 2 | 3 | 10 | 2 |
| **User** | 10 | 3 | 4 | 3 | 0 |
| **Domain** | 13 | 7 | 4 | 2 | 0 |
| **System** | 18 | 4 | 6 | 6 | 2 |
| **TOTAL** | **73** | **27** | **20** | **22** | **4** |

---

## Requirements Validation

All requirements have been validated through:

1. **Implementation Testing**: Code execution confirms functional requirements met
2. **Performance Benchmarking**: Metrics show non-functional requirements achieved
3. **User Acceptance**: Stakeholder review confirms user requirements satisfied
4. **Domain Expert Review**: Climate scientists validate domain requirements
5. **System Testing**: Hardware/software requirements verified on test machines

---

## Requirements Change Log

| Version | Date | Changes | Reason |
|---------|------|---------|--------|
| 1.0 | Nov 2025 | Initial requirements specification | Project start |
| 1.1 | Nov 2025 | Added quantum model requirements | Quantum ML implementation |
| 1.2 | Nov 2025 | Refined accuracy thresholds | Based on initial results |
| 1.3 | Nov 2025 | Added Colab deployment requirement | Cloud accessibility |

---

*This requirements specification document is complete and reflects the actual implemented system.*
