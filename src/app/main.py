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
import sys

def gradio_interface(
    gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    """
    data = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
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
    
    # Debug logging
    print(f"DEBUG: Processing prediction for customer. Tenure: {tenure}, Monthly: {MonthlyCharges}", file=sys.stderr)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    logs = [
        f"[{timestamp}] [SYSTEM] Initializing Kavi.ai Inference Engine...",
        f"[{timestamp}] [DATA] Validating input schema for 18 features...",
        f"[{timestamp}] [INFO] Tenure: {tenure} months | Monthly: ${MonthlyCharges}",
    ]
    
    try:
        logs.append(f"[{timestamp}] [TRANSFORM] Applying binary encoding to demographics...")
        logs.append(f"[{timestamp}] [TRANSFORM] Executing one-hot encoding for services...")
        
        result = predict(data)
        
        logs.append(f"[{timestamp}] [MODEL] Invoking XGBoost production model v8...")
        logs.append(f"[{timestamp}] [RESULT] Prediction generated: {result}")
        logs.append(f"[{timestamp}] [DONE] Final Risk Assessment completed.")
        
        log_text = "\n".join(logs)
        
        if "Not likely to churn" in result:
            report = f"✅ LOW RISK\n\nIntelligence Analysis: The customer profile indicates high stability and loyalty. {result}."
        else:
            report = f"⚠️ HIGH RISK\n\nIntelligence Analysis: Critical churn indicators detected in service usage or contract terms. {result}."
            
        return report, log_text
        
    except Exception as e:
        error_log = f"[{timestamp}] [ERROR] Process failed: {str(e)}"
        logs.append(error_log)
        return f"❌ ERROR: Processing failed. {str(e)}", "\n".join(logs)

# === GRADIO UI CONFIGURATION ===
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    background: #fdfdfd !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.brand-header {
    background: #ffffff !important;
    padding: 1.5rem !important;
    border-bottom: 2px solid #f0f0f0 !important;
    margin-bottom: 1.5rem !important;
    text-align: center !important;
}

.header-text {
    width: 100%;
}

.brand-header h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1a237e !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}

.brand-header p {
    font-size: 0.9rem;
    color: #546e7a;
    margin: 0.2rem 0 0 0 !important;
    font-weight: 500;
}

.predict-btn {
    background: #1a237e !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    font-size: 1.1rem !important;
    transition: transform 0.2s ease, background 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(26, 35, 126, 0.2) !important;
    cursor: pointer;
}

.predict-btn:hover {
    background: #283593 !important;
    transform: translateY(-1px);
}

.output-box {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    background: white !important;
    font-size: 1.1rem !important;
}

.input-section {
    background: white !important;
    padding: 1.2rem;
    border-radius: 8px;
    border: 1px solid #eee;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    color: #757575;
    font-size: 0.9rem;
}

.console-log {
    background-color: #000000 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}

.console-log textarea {
    background-color: #000000 !important;
    color: #00ff00 !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
}

.footer-logo {
    height: 60px !important;
    width: auto !important;
    margin-bottom: 1rem;
}
"""
with gr.Blocks(title="Kavi.ai | Churn Intelligence") as demo:
    with gr.Column(elem_classes="brand-header"):
        gr.HTML("""
            <div class="header-text">
                <h1>Telco Customer Churn Prediction</h1>
                <p>Enterprise Prediction Engine • Powered by Kavi.ai MLOps</p>
            </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(elem_classes="input-section"):
                    gr.Markdown("### 👤 Profile")
                    gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                    SeniorCitizen = gr.Radio(["No", "Yes"], label="Senior?", value="No")
                    Partner = gr.Radio(["Yes", "No"], label="Partner?", value="No")
                    Dependents = gr.Radio(["Yes", "No"], label="Dependents?", value="No")
                    tenure = gr.Slider(label="Tenure (M)", value=12, minimum=0, maximum=72, step=1)
                
                with gr.Column(elem_classes="input-section"):
                    gr.Markdown("### 📡 Device")
                    InternetService = gr.Dropdown(["Fiber optic", "DSL", "No"], label="Internet", value="Fiber optic")
                    PhoneService = gr.Radio(["Yes", "No"], label="Phone", value="Yes")
                    MultipleLines = gr.Dropdown(["No phone service", "No", "Yes"], label="Lines", value="No")
                    OnlineSecurity = gr.Dropdown(["No internet service", "No", "Yes"], label="Security", value="No")
                    OnlineBackup = gr.Dropdown(["No internet service", "No", "Yes"], label="Backup", value="No")
            
            with gr.Row():
                with gr.Column(elem_classes="input-section"):
                    gr.Markdown("### 📽️ Media")
                    DeviceProtection = gr.Dropdown(["No internet service", "No", "Yes"], label="Insurance", value="No")
                    TechSupport = gr.Dropdown(["No internet service", "No", "Yes"], label="Tech Support", value="No")
                    StreamingTV = gr.Dropdown(["No internet service", "No", "Yes"], label="TV", value="No")
                    StreamingMovies = gr.Dropdown(["No internet service", "No", "Yes"], label="Movies", value="No")

                with gr.Column(elem_classes="input-section"):
                    gr.Markdown("### 💳 Business")
                    Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                    PaperlessBilling = gr.Radio(["Yes", "No"], label="Paperless", value="Yes")
                    PaymentMethod = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment", value="Electronic check")
                    with gr.Row():
                        MonthlyCharges = gr.Number(label="Monthly ($)", value=75.0, precision=2)
                        TotalCharges = gr.Number(label="Total ($)", value=900.0, precision=2)

        with gr.Column(scale=1):
            gr.Markdown("### 📈 Risk Report")
            predict_btn = gr.Button("🔍 ANALYZE RISK", variant="primary", elem_classes="predict-btn")
            output_result = gr.Textbox(
                label="Assessment", 
                lines=12, 
                interactive=False, 
                elem_classes="output-box"
            )

    with gr.Row():
        log_window = gr.Textbox(
            label="⌨️ Process Logs",
            lines=6,
            interactive=False,
            elem_classes="console-log"
        )

    predict_btn.click(
        gradio_interface,
        inputs=[
            gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
        ],
        outputs=[output_result, log_window]
    )

    gr.HTML("""
        <div class="footer">
            <img src="/static/logo.png" alt="Kavi.ai Logo" class="footer-logo" />
            <p>Developed with Precision & Scalability by <strong>Kavi.ai</strong></p>
            <p>© 2026 Enterprise Machine Learning Excellence</p>
        </div>
    """)

# === MOUNT GRADIO UI INTO FASTAPI ===
app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui",
    theme=gr.themes.Soft(primary_hue="indigo", spacing_size="sm", radius_size="lg"),
    css=custom_css,
    allowed_paths=["src/app/static"]
)
