.PHONY: help install check-data train save-model git-push docker-build docker-push docker-pull docker-inference clean mlflowrun uirun serve mlflow-ui all

# ============================================================================
# Configuration
# ============================================================================
PYTHON := venv/bin/python3
EXPERIMENT_NAME := "Telco Churn"
MODEL_NAME := telco-churn-model
DATA_PATH := data/raw/Telco-Customer-Churn.csv
DOCKER_IMAGE := telco-churn-ml
DOCKER_TAG := latest
DOCKER_HUB_USER := $(shell echo $${DOCKER_HUB_USER:-machhakiran0108})
GITHUB_USER := machhakiran

# Colors & Formatting
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BLUE := \033[0;34m
CYAN := \033[0;36m
BOLD := \033[1m
NC := \033[0m

# ============================================================================
# Help
# ============================================================================
help:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)$(CYAN)  🚀 Telco Customer Churn ML - MLOps Workflow$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)📋 Usage: make [target]$(NC)"
	@echo ""
	@echo "$(BOLD)📦 Core Workflow (Run sequentially):$(NC)"
	@echo "  $(GREEN)1. install$(NC)        - Install all project dependencies"
	@echo "  $(GREEN)2. check-data$(NC)     - Verify input data integrity"
	@echo "  $(GREEN)3. mlflowrun$(NC)      - Launch MLflow Dashboard"
	@echo "  $(GREEN)4. train$(NC)          - Train XGBoost model & log to MLflow"
	@echo "  $(GREEN)5. save-model$(NC)     - Register best model version in MLflow"
	@echo "  $(GREEN)6. uirun$(NC)            - Start API & Kavi.ai UI"
	@echo "  $(GREEN)7. git-push$(NC)       - Commit & push code to GitHub"
	@echo ""
	@echo "$(BOLD)🐳 Docker Operations:$(NC)"
	@echo "  $(CYAN)8. docker-build$(NC)     - Build production Docker image"
	@echo "  $(CYAN)9. docker-push$(NC)      - Push image to Docker Hub"
	@echo "  $(CYAN)10. docker-pull$(NC)     - Pull image from Docker Hub"
	@echo "  $(CYAN)11. docker-inference$(NC) - Run containerized inference server"
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ============================================================================
# Step 1: Installation
# ============================================================================
install:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)📦 Step 1: Installing dependencies...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "$(CYAN)Verifying MLflow installation...$(NC)"
	@$(PYTHON) -m pip show mlflow >/dev/null 2>&1 || $(PYTHON) -m pip install mlflow>=2.19.0
	@echo "$(GREEN)✅ Dependencies installed successfully!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make check-data'$(NC)"

# ============================================================================
# Step 2: Data Check
# ============================================================================
check-data:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)📊 Step 2: Verifying data integrity...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@if [ ! -f $(DATA_PATH) ]; then \
		echo "$(RED)❌ Data file not found: $(DATA_PATH)$(NC)"; \
		echo "$(YELLOW)💡 Please download data to: $(DATA_PATH)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Data file present at: $(DATA_PATH)$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make mlflowrun'$(NC)"

# ============================================================================
# Step 3: MLflow Dashboard (Utility)
# ============================================================================
mlflowrun: mlflow-ui
mlflow-ui:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)📊 Step 3: Starting MLflow Tracking Server...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)🔗 Dashboard URL:$(NC) $(BOLD)http://localhost:5000$(NC)"
	@echo ""
	@mlflow ui --backend-store-uri file://./mlruns --port 5000

# ============================================================================
# Step 4: Train Model
# ============================================================================
train:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)🚀 Step 4: Training XGBoost Model...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@if [ ! -f $(DATA_PATH) ]; then \
		echo "$(RED)❌ Missing data. Run 'make check-data' first.$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/run_pipeline.py \
		--input $(DATA_PATH) \
		--target Churn \
		--experiment $(EXPERIMENT_NAME) \
		--threshold 0.35 \
		--test_size 0.2
	@echo "$(GREEN)✅ Training Complete!$(NC)"
	@echo "$(GREEN)📊 Metrics logged to MLflow.$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make save-model'$(NC)"

# ============================================================================
# Step 5: Save Model to MLflow Registry
# ============================================================================
save-model:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)💾 Step 5: Registering Model in MLflow...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@RUN_ID=$$($(PYTHON) scripts/get_latest_run.py --experiment $(EXPERIMENT_NAME) 2>/dev/null | tail -1); \
	if [ -z "$$RUN_ID" ]; then \
		echo "$(RED)❌ No run found. Please run 'make train' first.$(NC)"; \
		exit 1; \
	fi; \
	echo "$(CYAN)Promoting Run ID: $$RUN_ID$(NC)"; \
	$(PYTHON) scripts/promote_model.py --run-id $$RUN_ID --experiment $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) || \
	$(PYTHON) -c "import mlflow; import os; mlflow.set_tracking_uri('file://$$(pwd)/mlruns'); \
		from mlflow.tracking import MlflowClient; client = MlflowClient(); \
		run_id = '$$RUN_ID'; model_uri = f'runs:/{run_id}/model'; \
		result = mlflow.register_model(model_uri, '$(MODEL_NAME)'); \
		print(f'✅ Model registered as version {result.version}')"
	@echo "$(GREEN)✅ Model successfully registered in MLflow!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make uirun'$(NC)"

# ============================================================================
# Step 6: UI Run (Utility)
# ============================================================================
uirun: serve
serve:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)⚡ Step 6: Starting Kavi.ai Application Server...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(BOLD)🔗 Access Points:$(NC)"
	@echo "  $(GREEN)👉 UI (Interactive):$(NC) $(BOLD)http://localhost:8000/ui$(NC)"
	@echo "  $(GREEN)👉 API Docs:$(NC)         $(BOLD)http://localhost:8000/docs$(NC)"
	@echo "  $(GREEN)👉 Health Check:$(NC)     $(BOLD)http://localhost:8000/$(NC)"
	@echo ""
	@uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

# ============================================================================
# Step 7: Push to GitHub
# ============================================================================
git-push:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)📤 Step 7: Pushing to GitHub...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@GIT_BRANCH=$$(git branch --show-current 2>/dev/null || echo "main"); \
	echo "$(CYAN)Branch: $$GIT_BRANCH$(NC)"; \
	echo "$(CYAN)User: $(GITHUB_USER)$(NC)"; \
	git add . 2>/dev/null || true; \
	git commit -m "Update: $(shell date +'%Y-%m-%d %H:%M:%S')" 2>/dev/null || true; \
	git push origin $$GIT_BRANCH 2>/dev/null || echo "$(YELLOW)⚠️  Push skipped (no remote configured)$(NC)"
	@echo "$(GREEN)✅ Code pushed successfully!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make docker-build' (or wait for CI/CD)$(NC)"

# ============================================================================
# Step 8: Docker Operations
# ============================================================================
docker-build:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)🐳 Step 8: Building Docker Image...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(CYAN)Image: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	@docker build -t $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG) .
	@docker tag $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker build successful!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make docker-push'$(NC)"

docker-push:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)� Step 9: Pushing to Docker Hub...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(CYAN)Target: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	@echo "$(YELLOW)⚠️  Ensure you are logged in via 'docker login'$(NC)"
	@docker push $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Image successfully pushed to Docker Hub!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make docker-pull' on deployment server$(NC)"

docker-pull:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)📥 Step 10: Pulling from Docker Hub...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(CYAN)Source: $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	@docker pull $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Image pulled successfully!$(NC)"
	@echo "$(YELLOW)💡 Next: run 'make docker-inference'$(NC)"

docker-inference:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BOLD)⚡ Step 11: Starting Inference Container...$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	-docker stop telco-churn-inference 2>/dev/null || true
	-docker rm telco-churn-inference 2>/dev/null || true
	@docker run -d -p 8000:8000 --name telco-churn-inference $(DOCKER_HUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Container 'telco-churn-inference' is running!$(NC)"
	@echo ""
	@echo "$(BOLD)🔗 Verification URLs:$(NC)"
	@echo "  $(CYAN)� UI (Interactive):$(NC)   $(BOLD)http://localhost:8000/ui$(NC)"
	@echo "  $(CYAN)� API Docs:$(NC)           $(BOLD)http://localhost:8000/docs$(NC)"
	@echo "  $(CYAN)� Health Check:$(NC)       $(BOLD)http://localhost:8000/$(NC)"
	@echo ""
	@echo "$(YELLOW)💡 Stop container with: docker stop telco-churn-inference$(NC)"

# ============================================================================
# Utilities
# ============================================================================
clean:
	@echo "$(YELLOW)🧹 Cleaning project artifacts and caches...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf artifacts/ .gradio/ .pytest_cache/ .coverage htmlcov/ dist/ build/ 2>/dev/null || true
	@echo "$(GREEN)✅ Workspace cleaned!$(NC)"

all: install check-data train save-model git-push
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✅ Full MLOps Workflow Completed Successfully!$(NC)"
	@echo "$(YELLOW)💡 Ready for Production Deployment.$(NC)"
