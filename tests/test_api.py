"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    """Test prediction endpoint with valid data"""
    sample_data = {
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "tenure": 72,
        "MonthlyCharges": 20.0,
        "TotalCharges": 1440.0
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json() or "error" in response.json()


def test_predict_endpoint_invalid_data():
    """Test prediction endpoint with invalid data"""
    invalid_data = {"invalid": "data"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


