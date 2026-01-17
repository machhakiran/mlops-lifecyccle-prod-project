#!/usr/bin/env python3
"""Promote a model version to Production in MLflow Model Registry"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def promote_model_to_production(run_id: str, experiment_name: str = "Telco Churn", model_name: str = "telco-churn-model"):
    """Promote a model version to Production stage in MLflow Model Registry"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    client = MlflowClient(tracking_uri=mlruns_path)
    
    try:
        run = client.get_run(run_id)
        print(f"📦 Found run: {run_id}")
        
        # Register model if not already registered
        model_versions = client.search_model_versions(f"run_id='{run_id}'")
        if model_versions:
            version = model_versions[0].version
            print(f"✅ Model version {version} found in registry")
        else:
            print(f"📝 Registering model '{model_name}'...")
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            print(f"✅ Model registered as version {version}")
        
        # Promote to Production
        print(f"🚀 Promoting model version {version} to Production...")
        client.transition_model_version_stage(name=model_name, version=version, stage="Production")
        
        model_version = client.get_model_version(model_name, version)
        print(f"✅ Model version {version} promoted to {model_version.current_stage}")
        print(f"   Model URI: models:/{model_name}/Production")
        
        return version
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote model to Production in MLflow")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--experiment", type=str, default="Telco Churn", help="MLflow experiment name")
    parser.add_argument("--model-name", type=str, default="telco-churn-model", help="Model name in registry")
    args = parser.parse_args()
    promote_model_to_production(args.run_id, args.experiment, args.model_name)
