.PHONY: help install check-data train save-model git-push docker-build docker-push docker-pull docker-inference clean mlflow-ui

# ============================================================================
# Configuration
# ============================================================================
PYTHON := python3
EXPERIMENT_NAME := "Telco Churn"
MODEL_NAME := telco-churn-model
DATA_PATH := data/raw/Telco-Customer-Churn.csv
DOCKER_IMAGE := telco-churn-ml
DOCKER_TAG := latest
DOCKER_HUB_USER := $(shell echo $${DOCKER_HUB_USER:-machhakiran0108})
GITHUB_USER := machhakiran

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

# ============================================================================
# Help
# ============================================================================
help:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)  Telco Customer Churn ML - Clean Workflow$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)📦 Workflow Steps:$(NC)"
	@echo "  1. make install        - Install dependencies"
	@echo "  2. make check-data    - Verify data exists"
	@echo "  3. make train         - Train model (saves to MLflow)"
	@echo "  4. make save-model    - Save model to MLflow registry"
	@echo "  5. make git-push      - Push code to GitHub"
	@echo "  6. make docker-build  - Build Docker image"
	@echo "  7. make docker-push   - Push to Docker Hub"
	@echo "  8. make docker-pull  - Pull from Docker Hub"
	@echo "  9. make docker-inference - Run inference server"
	@echo ""
	@echo "$(YELLOW)🔧 Utilities:$(NC)"
	@echo "  make serve           - Start FastAPI server (port 8000)"
	@echo "  make mlflow-ui       - View MLflow experiments (port 5000)"
	@echo "  make clean           - Clean temporary files"
	@echo ""
	@echo "$(YELLOW)💡 Complete Workflow:$(NC)"
	@echo "  make install && make check-data && make train && make save-model && make git-push"
	@echo "  make docker-build && make docker-push"
	@echo "  make docker-pull && make docker-inference"
	@echo ""
	@echo "$(YELLOW)📝 Default Docker Hub User: machhakiran0108$(NC)"

# ============================================================================
# Step 1: Installation
# ============================================================================
install:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📦 Step 1: Installing dependencies...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Verifying MLflow installation...$(NC)"
	pip show mlflow >/dev/null 2>&1 || pip install mlflow>=2.19.0
	@echo "$(GREEN)✅ Installation complete!$(NC)"
	@echo "$(YELLOW)💡 Next: make check-data$(NC)"

# ============================================================================
# Step 2: Data Check
# ============================================================================
check-data:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📊 Step 2: Checking data...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@if [ ! -f $(DATA_PATH) ]; then \
		echo "$(RED)❌ Data file not found: $(DATA_PATH)$(NC)"; \
		echo "$(YELLOW)💡 Please place your data file at: $(DATA_PATH)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Data file found: $(DATA_PATH)$(NC)"
	@echo "$(YELLOW)💡 Next: make train$(NC)"

# ============================================================================
# Step 3: Train Model
# ============================================================================
train:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🚀 Step 3: Training model...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@if [ ! -f $(DATA_PATH) ]; then \
		echo "$(RED)❌ Data file not found. Run 'make check-data' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/run_pipeline.py \
		--input $(DATA_PATH) \
		--target Churn \
		--experiment $(EXPERIMENT_NAME) \
		--threshold 0.35 \
		--test_size 0.2
	@echo "$(GREEN)✅ Training completed! Model saved to MLflow$(NC)"
	@echo "$(YELLOW)💡 Next: make save-model$(NC)"

# ============================================================================
# Step 4: Save Model to MLflow Registry
# ============================================================================
save-model:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)💾 Step 4: Saving model to MLflow registry...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@RUN_ID=$$($(PYTHON) scripts/get_latest_run.py --experiment $(EXPERIMENT_NAME) 2>/dev/null | tail -1); \
	if [ -z "$$RUN_ID" ]; then \
		echo "$(RED)❌ No runs found. Train a model first with 'make train'$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)Saving run: $$RUN_ID$(NC)"; \
	$(PYTHON) scripts/promote_model.py --run-id $$RUN_ID --experiment $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) || \
	$(PYTHON) -c "import mlflow; import os; mlflow.set_tracking_uri('file://$$(pwd)/mlruns'); \
		from mlflow.tracking import MlflowClient; client = MlflowClient(); \
		run_id = '$$RUN_ID'; model_uri = f'runs:/{run_id}/model'; \
		result = mlflow.register_model(model_uri, '$(MODEL_NAME)'); \
		print(f'✅ Model registered as version {result.version}')"
	@echo "$(GREEN)✅ Model saved to MLflow registry$(NC)"
	@echo "$(YELLOW)💡 Next: make git-push$(NC)"

# ============================================================================
# Step 5: Push to GitHub
# ============================================================================
git-push:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📤 Step 5: Pushing code to GitHub...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@GIT_BRANCH=$$(git branch --show-current 2>/dev/null || echo "main"); \
	echo "$(GREEN)Pushing to branch: $$GIT_BRANCH$(NC)"; \
	echo "$(GREEN)GitHub User: $(GITHUB_USER)$(NC)"; \
	git add . 2>/dev/null || true; \
	git commit -m "Update: $(shell date +'%Y-%m-%d %H:%M:%S')" 2>/dev/null || true; \
	git push origin $$GIT_BRANCH 2>/dev/null || echo "$(YELLOW)⚠️  Git push skipped (not a git repo or no remote)$(NC)"
	@echo "$(GREEN)✅ Code pushed to GitHub$(NC)"
	@echo "$(YELLOW)💡 GitHub Actions will automatically build and push Docker image$(NC)"
	@echo "$(YELLOW)💡 Next: make docker-build (or wait for CI/CD)$(NC)"

# ============================================================================
# Step 6: Build Docker Image
# ============================================================================
docker-build:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🐳 Step 6: Building Docker image...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Building: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	docker build -t $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker image built$(NC)"
	@echo "$(YELLOW)💡 Next: make docker-push$(NC)"

# ============================================================================
# Step 7: Push to Docker Hub
# ============================================================================
docker-push:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📤 Step 7: Pushing Docker image to Docker Hub...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Pushing: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	@echo "$(YELLOW)💡 Make sure you're logged in: docker login$(NC)"
	docker push $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker image pushed to Docker Hub$(NC)"
	@echo "$(YELLOW)💡 Next: make docker-pull (on another machine)$(NC)"

# ============================================================================
# Step 8: Pull from Docker Hub
# ============================================================================
docker-pull:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📥 Step 8: Pulling Docker image from Docker Hub...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Pulling: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	docker pull $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker image pulled$(NC)"
	@echo "$(YELLOW)💡 Next: make docker-inference$(NC)"

# ============================================================================
# Step 9: Run Inference Server
# ============================================================================
docker-inference:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🚀 Step 9: Running inference server from Docker Hub...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Running: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	-docker stop telco-churn-inference 2>/dev/null || true
	-docker rm telco-churn-inference 2>/dev/null || true
	docker run -d -p 8000:8000 --name telco-churn-inference $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Inference server started$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📍 API: http://localhost:8000$(NC)"
	@echo "$(GREEN)📍 Docs: http://localhost:8000/docs$(NC)"
	@echo "$(GREEN)📍 UI: http://localhost:8000/ui$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(YELLOW)💡 Stop with: docker stop telco-churn-inference && docker rm telco-churn-inference$(NC)"

# ============================================================================
# Utilities
# ============================================================================
serve:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🚀 Starting FastAPI server...$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)📍 Health Check: http://localhost:8000/$(NC)"
	@echo "$(GREEN)📍 API Docs: http://localhost:8000/docs$(NC)"
	@echo "$(GREEN)📍 Gradio UI: http://localhost:8000/ui$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	@echo "$(GREEN)📊 Starting MLflow UI...$(NC)"
	@echo "$(YELLOW)MLflow UI: http://localhost:5000$(NC)"
	mlflow ui --backend-store-uri file://./mlruns --port 5000

clean:
	@echo "$(GREEN)🧹 Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml dist/ build/ 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

# ============================================================================
# Complete Workflow (All Steps)
# ============================================================================
all: install check-data train save-model git-push
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✅ Complete workflow finished!$(NC)"
	@echo "$(YELLOW)💡 Next: make docker-build && make docker-push$(NC)"
	@echo "$(YELLOW)💡 Or wait for GitHub Actions to build and push automatically$(NC)"
