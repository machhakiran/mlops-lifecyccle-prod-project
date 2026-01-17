# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-17

### Added
- Initial release of Telco Customer Churn ML project
- End-to-end ML pipeline with data preprocessing, feature engineering, and model training
- XGBoost classifier for churn prediction
- FastAPI REST API for model serving
- Gradio web UI for interactive predictions
- MLflow integration for experiment tracking
- Docker containerization for deployment
- CI/CD pipeline with GitHub Actions
- Production-ready project structure

### Features
- Data quality validation using Great Expectations
- Feature engineering pipeline with binary and one-hot encoding
- Hyperparameter optimization with Optuna
- Model versioning and artifact management
- Health check endpoints for load balancer integration
- Comprehensive API documentation with OpenAPI/Swagger

### Infrastructure
- Docker containerization
- AWS ECS Fargate deployment support
- Application Load Balancer (ALB) integration
- CloudWatch logging support
- Security group configuration

---

## [Unreleased]

### Planned
- Model monitoring and drift detection
- A/B testing framework
- Enhanced documentation
