#!/usr/bin/env python3
"""Evaluate a trained model from MLflow and log metrics to MLflow"""
import os
import sys
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import json

def evaluate_model_from_mlflow(run_id: str, test_data_path: str, experiment_name: str = "Telco Churn", log_to_mlflow: bool = True):
    """Evaluate a model loaded from MLflow and optionally log metrics to MLflow"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    client = MlflowClient(tracking_uri=mlruns_path)
    
    try:
        train_run = client.get_run(run_id)
        print(f"📦 Evaluating model from run: {run_id}")
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded from {model_uri}")
        
        # Load and preprocess test data
        print(f"📊 Loading test data from {test_data_path}...")
        df_test = pd.read_csv(test_data_path)
        
        target_col = "Churn"
        if target_col not in df_test.columns:
            print(f"❌ Target column '{target_col}' not found", file=sys.stderr)
            sys.exit(1)
        
        # Apply preprocessing
        sys.path.append(project_root)
        from src.data.preprocess import preprocess_data
        from src.features.build_features import build_features
        
        df_test = preprocess_data(df_test)
        df_test = build_features(df_test, target_col=target_col)
        
        # Convert boolean to int
        for c in df_test.select_dtypes(include=["bool"]).columns:
            df_test[c] = df_test[c].astype(int)
        
        # Prepare features
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]
        threshold = float(train_run.data.params.get("threshold", 0.35))
        
        # Make predictions
        print(f"🔮 Making predictions with threshold={threshold}...")
        proba = model.predict(X_test)
        if len(proba.shape) > 1:
            proba = proba[:, 1]
        
        y_pred = (proba >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f} | ROC AUC: {roc_auc:.4f}")
        print(f"   TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
        
        # Log to MLflow if requested
        if log_to_mlflow:
            print(f"\n💾 Logging evaluation metrics to MLflow...")
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"evaluation-{run_id[:8]}"):
                mlflow.set_tag("run_type", "evaluation")
                mlflow.set_tag("training_run_id", run_id)
                
                # Log parameters
                mlflow.log_param("model", train_run.data.params.get("model", "xgboost"))
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("evaluation_data_path", test_data_path)
                
                # Log metrics
                mlflow.log_metric("eval_precision", precision)
                mlflow.log_metric("eval_recall", recall)
                mlflow.log_metric("eval_f1", f1)
                mlflow.log_metric("eval_roc_auc", roc_auc)
                mlflow.log_metric("eval_tp", int(tp))
                mlflow.log_metric("eval_tn", int(tn))
                mlflow.log_metric("eval_fp", int(fp))
                mlflow.log_metric("eval_fn", int(fn))
                mlflow.log_metric("eval_test_samples", len(y_test))
                
                # Save artifacts
                artifacts_dir = os.path.join(project_root, "artifacts")
                os.makedirs(artifacts_dir, exist_ok=True)
                
                report = classification_report(y_test, y_pred, output_dict=True)
                report_path = os.path.join(artifacts_dir, "evaluation_report.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_path, "evaluation")
                
                cm_dict = {
                    "confusion_matrix": cm.tolist(),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
                }
                cm_path = os.path.join(artifacts_dir, "confusion_matrix.json")
                with open(cm_path, "w") as f:
                    json.dump(cm_dict, f, indent=2)
                mlflow.log_artifact(cm_path, "evaluation")
                
                eval_run_id = mlflow.active_run().info.run_id
                print(f"✅ Evaluation metrics logged to MLflow run: {eval_run_id}")
        
        return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model from MLflow and log metrics")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow training run ID")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--experiment", type=str, default="Telco Churn", help="MLflow experiment name")
    parser.add_argument("--no-log", action="store_true", help="Don't log metrics to MLflow")
    args = parser.parse_args()
    evaluate_model_from_mlflow(args.run_id, args.test_data, args.experiment, log_to_mlflow=not args.no_log)
