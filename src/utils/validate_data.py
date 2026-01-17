from typing import Tuple, List
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation...")
    
    failed_expectations = []
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            failed_expectations.append(f"Missing column: {col}")
    
    if failed_expectations:
        print(f"❌ Schema validation FAILED: Missing columns")
        return False, failed_expectations
    
    # Check for null values in critical columns
    if df["customerID"].isnull().any():
        failed_expectations.append("customerID has null values")
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values
    if not df["gender"].isin(["Male", "Female"]).all():
        failed_expectations.append("gender contains invalid values")
    
    # Yes/No fields must have valid values
    yes_no_fields = ["Partner", "Dependents", "PhoneService"]
    for field in yes_no_fields:
        if not df[field].isin(["Yes", "No"]).all():
            failed_expectations.append(f"{field} contains invalid values (expected Yes/No)")
    
    # Contract types must be valid
    valid_contracts = ["Month-to-month", "One year", "Two year"]
    if not df["Contract"].isin(valid_contracts).all():
        failed_expectations.append("Contract contains invalid values")
    
    # Internet service types
    valid_internet = ["DSL", "Fiber optic", "No"]
    if not df["InternetService"].isin(valid_internet).all():
        failed_expectations.append("InternetService contains invalid values")
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Convert numeric columns to proper types (handle string values)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if df[col].dtype == 'object':
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Tenure must be non-negative
    if (df["tenure"] < 0).any():
        failed_expectations.append("tenure has negative values")
    
    # Monthly charges must be positive
    if (df["MonthlyCharges"] < 0).any():
        failed_expectations.append("MonthlyCharges has negative values")
    
    # Total charges should be non-negative (excluding NaN)
    if (df["TotalCharges"].dropna() < 0).any():
        failed_expectations.append("TotalCharges has negative values")
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months)
    if (df["tenure"] > 120).any():
        failed_expectations.append("tenure exceeds reasonable maximum (120 months)")
    
    # Monthly charges should be within reasonable range
    if (df["MonthlyCharges"] > 200).any():
        failed_expectations.append("MonthlyCharges exceeds reasonable maximum ($200)")
    
    # No missing values in critical numeric features
    if df["tenure"].isnull().any():
        failed_expectations.append("tenure has null values")
    
    if df["MonthlyCharges"].isnull().any():
        failed_expectations.append("MonthlyCharges has null values")
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (with 5% tolerance)
    consistency_check = df["TotalCharges"] >= df["MonthlyCharges"]
    if consistency_check.mean() < 0.95:
        failed_expectations.append("TotalCharges < MonthlyCharges in more than 5% of records")
    
    # === PROCESS RESULTS ===
    total_checks = 15  # Number of validation checks performed
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks
    
    if not failed_expectations:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
        return True, []
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
        return False, failed_expectations

