import numpy as np
import pandas as pd
import glob
import os

def load_and_prepare_climate_data():
    """
    Load climate radar data from multiple CSV files and prepare for ML models
    Returns: X (features), y (binary labels for extreme weather detection)
    """
    print("="*70)
    print("LOADING CLIMATE DATASET")
    print("="*70)
    
    # Path to data files
    data_path = "data/extracted_data/extracted_data/*.csv"
    csv_files = glob.glob(data_path)
    
    print(f"\n[INFO] Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return None, None
    
    # Load all CSV files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    # Combine all dataframes
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset shape: {data.shape}")
    print(f"✓ Total samples: {len(data)}")
    
    # Display column information
    print(f"\n[INFO] Columns in dataset:")
    print(data.columns.tolist())
    
    # Select relevant numerical features for ML
    feature_columns = [
        'altitude', 'azimuth', 'elevation', 'range_min', 'range_max',
        'DBZ_mean', 'DBZ_max', 'DBZ_min', 'DBZ_std', 'DBZ_valid_count',
        'VEL_mean', 'VEL_max', 'VEL_min', 'VEL_std', 'VEL_valid_count',
        'WIDTH_mean', 'WIDTH_max', 'WIDTH_min', 'WIDTH_std', 'WIDTH_valid_count',
        'ZDR_mean', 'ZDR_max', 'ZDR_min', 'ZDR_std', 'ZDR_valid_count',
        'PHIDP_mean', 'PHIDP_max', 'PHIDP_min', 'PHIDP_std', 'PHIDP_valid_count',
        'RHOHV_mean', 'RHOHV_max', 'RHOHV_min', 'RHOHV_std', 'RHOHV_valid_count'
    ]
    
    # Extract features and convert to numeric (handle any string values)
    X = data[feature_columns].apply(pd.to_numeric, errors='coerce').values
    
    print(f"\n[INFO] Features extracted: {X.shape}")
    print(f"[INFO] Feature names: {len(feature_columns)} features")
    
    # Create binary labels based on extreme weather conditions
    # Label 1: Extreme weather (high DBZ reflectivity indicating heavy precipitation)
    # Label 0: Normal weather
    threshold_dbz = 30  # dBZ threshold for severe weather
    
    # Convert DBZ_max to numeric and handle any non-numeric values
    dbz_max_numeric = pd.to_numeric(data['DBZ_max'], errors='coerce')
    y = (dbz_max_numeric > threshold_dbz).astype(int).values
    
    print(f"\n[INFO] Labels created based on DBZ_max > {threshold_dbz} dBZ")
    print(f"[INFO] Class distribution:")
    print(f"  Normal weather (0): {np.sum(y == 0)} samples ({100*np.sum(y == 0)/len(y):.1f}%)")
    print(f"  Extreme weather (1): {np.sum(y == 1)} samples ({100*np.sum(y == 1)/len(y):.1f}%)")
    
    # Handle missing values
    if np.any(np.isnan(X)):
        print(f"\n⚠️  Found {np.sum(np.isnan(X))} missing values, filling with column means...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print("✓ Missing values handled")
    
    # Handle infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✓ Final dataset ready:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Feature statistics:")
    print(f"    Mean: {X.mean():.2f}")
    print(f"    Std: {X.std():.2f}")
    print(f"    Min: {X.min():.2f}")
    print(f"    Max: {X.max():.2f}")
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_prepare_climate_data()
    if X is not None:
        # Save processed data
        np.save('X_data.npy', X)
        np.save('y_labels.npy', y)
        print(f"\n✓ Saved processed data:")
        print(f"  X_data.npy: {X.shape}")
        print(f"  y_labels.npy: {y.shape}")
