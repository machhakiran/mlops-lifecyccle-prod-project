"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
MODEL_NAME = "telco-churn-model"
MODEL_STAGE = "Production"

# Initialize MLflow tracking
import mlflow
from mlflow.tracking import MlflowClient
tracking_uri = f"file://{os.getcwd()}/mlruns"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

try:
    # 1. Find the Production version in the registry
    versions = client.get_latest_versions(MODEL_NAME, [MODEL_STAGE])
    if not versions:
        raise Exception(f"No model version found for {MODEL_NAME} in {MODEL_STAGE} stage")
    
    prod_version = versions[0]
    run_id = prod_version.run_id
    print(f"📦 Found Production model (v{prod_version.version}) from run: {run_id}")
    
    # 2. Download artifacts: the model folder AND the feature metadata
    # We download the model folder specifically for pyfunc loading
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # We also need the feature columns text file which is at the run's root artifacts
    # download_artifacts returns the local path to the downloaded file/folder
    MODEL_DIR = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    FEATURE_FILE = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_columns.txt")
    
    print(f"✅ Model and features loaded from run {run_id}")

except Exception as registry_error:
    print(f"⚠️ Registry load failed: {registry_error}. Falling back to local scans...")
    try:
        import glob
        # Try to find any local model artifact
        model_paths = glob.glob("./mlruns/*/*/artifacts/model") or glob.glob("./mlruns/*/*/models/*/artifacts")
        if not model_paths:
            raise Exception("No local model artifacts found.")
        
        MODEL_DIR = max(model_paths, key=os.path.getmtime)
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        
        # Look for feature_columns.txt in same or parent dir
        FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.txt")
        if not os.path.exists(FEATURE_FILE):
            FEATURE_FILE = os.path.join(os.path.dirname(MODEL_DIR), "feature_columns.txt")
            
        if not os.path.exists(FEATURE_FILE):
             raise Exception("feature_columns.txt not found locally.")
             
        print(f"✅ Fallback: Loaded model from {MODEL_DIR}")
    except Exception as fallback_error:
        print(f"❌ CRITICAL ERROR: Could not load model. Run 'make train' then 'make save-model'.")
        raise fallback_error

# === FEATURE SCHEMA LOADING ===
try:
    with open(FEATURE_FILE) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Successfully loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise Exception(f"Failed to load feature columns metadata: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service
