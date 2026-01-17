import argparse
import mlflow
import shutil
import os
import sys
from mlflow.tracking import MlflowClient

# Set up basic logging
print("Preparing Docker build context...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment Name")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    args = parser.parse_args()

    # Set MLflow tracking URI to local mlruns if not set
    if not os.environ.get("MLFLOW_TRACKING_URI"):
            # Assuming script is run from project root or we find mlruns relative to script?
            # Better to expect it to be run from project root (Makefile does this)
            cwd = os.getcwd()
            mlruns_path = os.path.join(cwd, "mlruns")
            if os.path.exists(mlruns_path):
                uri = "file://" + mlruns_path
                mlflow.set_tracking_uri(uri)
                print(f"Tracking URI set to: {uri}")
            else:
                print("Warning: mlruns directory not found in current path.")

    client = MlflowClient()
    
    # 1. Get Experiment
    print(f"Searching for experiment: {args.experiment}")
    exp = client.get_experiment_by_name(args.experiment)
    if not exp:
        print(f"Error: Experiment '{args.experiment}' not found.")
        sys.exit(1)
        
    # 2. Get Latest Run
    print(f"Experiment ID: {exp.experiment_id}")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Error: No runs found for experiment.")
        sys.exit(1)
        
    run = runs[0]
    run_id = run.info.run_id
    print(f"Latest Run ID: {run_id} (Status: {run.info.status})")
    
    # 3. Download Artifacts using MLflow API (Reliable)
    target_dir = args.output
    if os.path.exists(target_dir):
        print(f"Cleaning existing target directory: {target_dir}")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
        
    print(f"Downloading all artifacts to: {target_dir}")
    try:
        # download_artifacts handles the source logic (file, s3, etc)
        client.download_artifacts(run_id, "", target_dir)
        print("✅ Success: Artifacts downloaded.")
        
        # Verify if 'model' folder exists
        if not os.path.exists(os.path.join(target_dir, "model")):
            print("⚠️  Warning: 'model' folder not found in artifacts. Inference might fail if not checked.")
            
    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
