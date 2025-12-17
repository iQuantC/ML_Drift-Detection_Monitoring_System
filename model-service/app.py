import os
import time
import logging
from typing import List, Dict
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response
import requests

from drift_detector import DriftDetector
from data_generator import DataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Model with Drift Monitoring")

# Configuration
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
CHECK_INTERVAL = 60  # seconds

# Initialize data generator
data_gen = DataGenerator(n_features=5)

# Generate reference data (training data)
reference_data = data_gen.generate_reference_data(n_samples=1000)
feature_names = data_gen.get_feature_names()

# Initialize drift detector
drift_detector = DriftDetector(reference_data, feature_names)

# Store recent predictions for drift detection
recent_predictions = []
MAX_RECENT_SAMPLES = 200

# Prometheus metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['prediction_class']
)

drift_score_gauge = Gauge(
    'model_drift_score',
    'Data drift score by feature',
    ['feature', 'method']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds'
)

drift_alert_counter = Counter(
    'drift_alerts_total',
    'Total number of drift alerts sent',
    ['severity']
)

feature_mean_gauge = Gauge(
    'model_feature_mean',
    'Mean value of features in recent data',
    ['feature']
)

feature_std_gauge = Gauge(
    'model_feature_std',
    'Standard deviation of features in recent data',
    ['feature']
)


# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: float
    prediction_class: str
    drift_detected: bool
    drift_scores: Dict[str, float]


class DriftStatus(BaseModel):
    overall_drift: str
    feature_drift_scores: Dict[str, float]
    threshold: float
    timestamp: float


def send_slack_alert(message: str, severity: str = "warning"):
    """Send alert to Slack"""
    if not SLACK_WEBHOOK_URL:
        logger.warning("Slack webhook URL not configured")
        return
    
    color_map = {
        "info": "#36a64f",
        "warning": "#ff9900",
        "critical": "#ff0000"
    }
    
    slack_data = {
        "attachments": [{
            "color": color_map.get(severity, "#808080"),
            "title": f"Drift Alert - {severity.upper()}",
            "text": message,
            "footer": "ML Drift Monitoring System",
            "ts": int(time.time())
        }]
    }
    
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_data,
            timeout=5
        )
        if response.status_code == 200:
            logger.info(f"Slack alert sent: {severity}")
            drift_alert_counter.labels(severity=severity).inc()
        else:
            logger.error(f"Failed to send Slack alert: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending Slack alert: {e}")


def check_and_alert_drift(drift_scores: Dict[str, float]):
    """Check drift scores and send alerts if threshold exceeded"""
    for feature, score in drift_scores.items():
        if score > DRIFT_THRESHOLD:
            status, severity = drift_detector.get_drift_status(score)
            message = (
                f" Drift detected in feature '{feature}'\n"
                f"Drift Score: {score:.4f}\n"
                f"Threshold: {DRIFT_THRESHOLD}\n"
                f"Status: {status}\n"
                f"Action: Review model performance and consider retraining"
            )
            send_slack_alert(message, severity)
            logger.warning(f"Drift alert sent for {feature}: {score:.4f}")


def update_drift_metrics():
    """Update drift metrics based on recent predictions"""
    if len(recent_predictions) < 50:
        logger.info("Not enough samples for drift detection")
        return
    
    # Convert to numpy array
    current_data = np.array(recent_predictions[-200:])
    
    # Detect drift using KS test
    ks_drift_scores = drift_detector.detect_drift_ks(current_data)
    
    # Detect drift using PSI
    psi_drift_scores = drift_detector.detect_drift_psi(current_data)
    
    # Update Prometheus metrics
    for feature, score in ks_drift_scores.items():
        drift_score_gauge.labels(feature=feature, method='ks_test').set(score)
    
    for feature, score in psi_drift_scores.items():
        drift_score_gauge.labels(feature=feature, method='psi').set(score)
    
    # Update feature statistics
    for i, feature in enumerate(feature_names):
        feature_data = current_data[:, i]
        feature_mean_gauge.labels(feature=feature).set(np.mean(feature_data))
        feature_std_gauge.labels(feature=feature).set(np.std(feature_data))
    
    # Check for drift and send alerts
    check_and_alert_drift(ks_drift_scores)
    
    logger.info(f"Drift metrics updated. Recent samples: {len(recent_predictions)}")


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "service": "ML Model with Drift Monitoring",
        "status": "running",
        "drift_threshold": DRIFT_THRESHOLD,
        "recent_samples": len(recent_predictions)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make prediction and monitor for drift"""
    start_time = time.time()
    
    # Convert to numpy array
    features = np.array(request.features)
    
    # Validate input
    if len(features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features, got {len(features)}"
        }
    
    # Make prediction
    prediction = data_gen.simulate_prediction(features)
    prediction_class = "positive" if prediction > 0.5 else "negative"
    
    # Record metrics
    prediction_counter.labels(prediction_class=prediction_class).inc()
    prediction_latency.observe(time.time() - start_time)
    
    # Store for drift detection
    recent_predictions.append(features)
    if len(recent_predictions) > MAX_RECENT_SAMPLES:
        recent_predictions.pop(0)
    
    # Check drift periodically
    drift_scores = {}
    drift_detected = False
    if len(recent_predictions) >= 50:
        current_data = np.array(recent_predictions[-100:])
        drift_scores = drift_detector.detect_drift_ks(current_data)
        drift_detected = any(score > DRIFT_THRESHOLD for score in drift_scores.values())
        
        # Update metrics in background
        background_tasks.add_task(update_drift_metrics)
    
    return PredictionResponse(
        prediction=float(prediction),
        prediction_class=prediction_class,
        drift_detected=drift_detected,
        drift_scores=drift_scores
    )


@app.get("/drift/status", response_model=DriftStatus)
def get_drift_status():
    """Get current drift status"""
    if len(recent_predictions) < 50:
        return DriftStatus(
            overall_drift="insufficient_data",
            feature_drift_scores={},
            threshold=DRIFT_THRESHOLD,
            timestamp=time.time()
        )
    
    current_data = np.array(recent_predictions[-200:])
    drift_scores = drift_detector.detect_drift_ks(current_data)
    
    max_drift = max(drift_scores.values()) if drift_scores else 0
    status, _ = drift_detector.get_drift_status(max_drift)
    
    return DriftStatus(
        overall_drift=status,
        feature_drift_scores=drift_scores,
        threshold=DRIFT_THRESHOLD,
        timestamp=time.time()
    )


@app.post("/simulate/drift")
def simulate_drift(severity: float = 0.5, n_requests: int = 100):
    """
    Simulate drift by generating predictions with drifted data
    
    Args:
        severity: Drift severity (0-1)
        n_requests: Number of requests to simulate
    """
    logger.info(f"Simulating drift with severity {severity}")
    
    # Generate drifted data
    drifted_data = data_gen.generate_production_data(
        n_samples=n_requests,
        drift_severity=severity,
        drift_features=[0, 1, 2]  # Apply drift to first 3 features
    )
    
    # Make predictions and store
    for features in drifted_data:
        prediction = data_gen.simulate_prediction(features)
        prediction_class = "positive" if prediction > 0.5 else "negative"
        prediction_counter.labels(prediction_class=prediction_class).inc()
        
        recent_predictions.append(features)
        if len(recent_predictions) > MAX_RECENT_SAMPLES:
            recent_predictions.pop(0)
    
    # Update drift metrics
    update_drift_metrics()
    
    return {
        "status": "drift_simulated",
        "severity": severity,
        "n_requests": n_requests,
        "message": f"Generated {n_requests} predictions with drift severity {severity}"
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
