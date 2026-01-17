"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 


# === STATIC FILES ===
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    
    # Return formatted result with emoji for better UX
    if "Not likely to churn" in result:
        return f"✅ Low Risk: {result}"
    else:
        return f"⚠️ High Risk: {result}"

# === GRADIO UI CONFIGURATION ===
# Custom CSS for Kavi.ai branding
custom_css = """
.gradio-container {background-color: #f8f9fa}
.brand-header {text-align: center; margin-bottom: 20px;}
.brand-header img {max-width: 150px; margin: 0 auto;}
.brand-header h1 {color: #4a148c; margin-top: 10px; font-weight: bold;}
.predict-btn {background: linear-gradient(90deg, #4a148c 0%, #01579b 100%); color: white; border: none;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css, title="Kavi.ai Churn Predictor") as demo:
    with gr.Column(elem_classes="brand-header"):
        gr.HTML("""
            <div class="brand-header">
                <img src="/static/logo.png" alt="Kavi.ai Logo" />
                <h1>Kavi.ai Operations</h1>
                <p>Telco Customer Churn Prediction System</p>
            </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("👤 Customer Demographics", open=True):
                gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
                tenure = gr.Slider(label="Tenure (months)", value=1, minimum=0, maximum=100, step=1)
        
        with gr.Column(scale=1):
            with gr.Accordion("📡 Services & Usage", open=True):
                InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🛡️ Support & Security", open=False):
                OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")

        with gr.Column(scale=1):
            with gr.Accordion("💳 Billing & Contract", open=True):
                Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                PaymentMethod = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check")
                MonthlyCharges = gr.Number(label="Monthly Charges ($)", value=85.0)
                TotalCharges = gr.Number(label="Total Charges ($)", value=85.0)

    with gr.Row():
        predict_btn = gr.Button("🔮 Predict Churn Risk", variant="primary", elem_classes="predict-btn")
        
    output_result = gr.Textbox(label="Prediction Result", lines=1, interactive=False)
    
    predict_btn.click(
        gradio_interface,
        inputs=[
            gender, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
        ],
        outputs=output_result
    )

    gr.Markdown("### 💡 Powered by Kavi.ai MLOps Pipeline")

# === MOUNT GRADIO UI INTO FASTAPI ===
app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui",
    allowed_paths=["src/app/static"]
)
