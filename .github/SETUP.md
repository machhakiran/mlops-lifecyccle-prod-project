# GitHub Actions Setup

## Required Secrets

Add these secrets to your GitHub repository settings (Settings → Secrets and variables → Actions):

### Docker Hub Secrets
- **Name:** `DOCKER_HUB_USERNAME`
- **Value:** `machhakiran0108`

- **Name:** `DOCKER_HUB_TOKEN`
- **Value:** Your Docker Hub access token (create at https://hub.docker.com/settings/security)
- **Note:** Keep your token secure! Never commit it to git.

### Optional Secrets
- **Name:** `CODECOV_TOKEN`
- **Value:** Your Codecov token (optional, for coverage reporting)

## How to Get Docker Hub Token

1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Give it a name (e.g., "github-actions")
4. Copy the token
5. Add it as `DOCKER_HUB_TOKEN` secret in GitHub

## Repository Information

- **GitHub User:** machhakiran
- **Docker Hub User:** machhakiran0108
- **Docker Image:** machhakiran0108/telco-churn-ml

## CI/CD Workflow

When you push to GitHub:
1. Tests run automatically
2. Code quality checks run
3. Docker image is built
4. Docker image is pushed to Docker Hub as `machhakiran0108/telco-churn-ml:latest`

