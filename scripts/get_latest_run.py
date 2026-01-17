#!/usr/bin/env python3
"""Get the latest MLflow run ID from an experiment"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def get_latest_run_id(experiment_name: str = "Telco Churn"):
    """Get the latest run ID from an MLflow experiment"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    client = MlflowClient(tracking_uri=mlruns_path)
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found", file=sys.stderr)
            sys.exit(1)
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"❌ No runs found in experiment '{experiment_name}'", file=sys.stderr)
            sys.exit(1)
        
        run_id = runs[0].info.run_id
        print(run_id)
        return run_id
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get latest MLflow run ID")
    parser.add_argument("--experiment", type=str, default="Telco Churn", help="MLflow experiment name")
    args = parser.parse_args()
    get_latest_run_id(args.experiment)
