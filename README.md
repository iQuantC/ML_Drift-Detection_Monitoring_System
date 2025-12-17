# Drift Detection Monitoring System - Prometheus, Grafana, AlertManager, and Slack
In this project, we give a practical, end-to-end MLOps project that detects data / concept drift, exports drift metrics to Prometheus, visualizes & alerts in Grafana, Alertmanager, and Slack. 

## GOAL
1. Model Service (FAstAPI):     Makes predictions, detects drift, and exposes "/metrics" endpoint.
2. Prometheus:                  Comprehensively collects, stores metrics & evaluates alert rules.
3. Grafana:                     Visualizes metrics on an Interactive UI (dashboard). 
4. Real-time Drift Detection:   Statistical Methods (Kolmogorov Smirnov(KS test)-Numerical data, Population Stability Index(PSI)-Categorical data)
5. Alertmanager:                Sends alerts to Slack when drift is detected. 
6. Slack:                       Messaging platform for receiving alerts (on phone, laptop, etc.) when drift exceeds threshold.
7. Docker Compose:              For easy deployment of Docker Containers for the Services.
 


## Project Structure
Create the necessary files and directories in the project root directory:
```sh
mkdir -p drift-monitoring/{prometheus,grafana/provisioning/{datasources,dashboards},model-service,alertmanager}
cd drift-monitoring
```


## Model Service (Python/FastAPI)
1. app.py:              Main API with endpoints
2. drift_detector.py:   KS Test & PSI algorithms
3. data_generator.py:   Synthetic data simulation
4. Real-time drift detection


### Model-Service API Endpoints
Model Service:

1. GET /                - Service status
2. POST /predict        - Make prediction
3. GET /drift/status    - Current drift status
4. POST /simulate/drift - Simulate drift for testing
5. GET /metrics         - Prometheus metrics
6. GET /health          - Health check


```sh
touch .env docker-compose.yml
```

## Build and Start the Services
```sh
# Build and start all services
docker-compose up -d --build

# Wait for all services to be ready (30 seconds)
sleep 30

# Check running docker-compose processes
docker-compose ps

# Check logs
docker-compose logs -f

# Check logs of individual service (model-service, prometheus, grafana, alertmanager)
docker-compose logs -f model-service

# Stop the running docker processes
docker-compose down -v

# Remove everything including images
docker-compose down -v --rmi all
```


## Verify Service Health
```sh
# Check model service
curl http://localhost:8000/health

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health
```


## Access the Services
```sh
# Model Service API
http://localhost:8000

# Prometheus
http://localhost:9090

# Grafana (username: admin, password: admin)
http://localhost:3000

# Alertmanager
http://localhost:9093
```

## Setup Slack
Install, create account and sign-in to your Slack Account. 

```sh
sudo snap install slack --classic
```

### Create a Slack Incoming Webhook
1. Go to your Slack workspace. Click File - Settings & Administration - Manage apps.
2. In the Search bar on the top right, search and open "Incoming Webhooks".
3. Click "Add to Slack".
4. Scroll down to "Post to Channel" and select the channel where you want to post alerts (or Create a New Channel).
5. Click "Add Incoming Webhooks Integration".
6. Copy & Save the Webhook URL, e.g.: https://hooks.slack.com/services/T09NYD7D30R/B09NHJYA5GE/u2he99h3f79h23hy9rK


### Configure Grafana to Use the Slack Webhook
1. Add SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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uZj to the .env file.
2. Add slack_api_url: 'https://hooks.slack.com/services/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Zj' to the alertmanager.yml.


## Generate Normal Traffic 
We will generate normal traffic from the ML Model using the "test_normal.sh" script to make some predictions.

```sh
# Make the shell script executable and run it
chmod +x test_normal.sh
```

```sh
./test_normal.sh
```


## Verify Metrics in Prometheus
On the Prometheus page, try the following queries: 

```sh
model_drift_score
model_drift_score{method="ks_test"}
model_drift_score{method="psi"}

# Should show firing alerts
ALERTS{alertstate="firing"}

# Should show specific drift alerts
ALERTS{alertname="DataDriftDetected"}
```

Verify on Grafana Dashboard as well. 

## Grafana Dashboard Includes the following
1. Drift Scores by Features:    Bar guage showing current drift
2. Drift Score Over Time:       Time series of drift evolution
3. Prediction Rate:             Predictions per second
4. Drift Alerts:                Counts of alerts in last hour
5. Total Predictions:           Cummulative count of predictions
6. Feature Means:               Statistical tracking
7. Feature Std Deviations:      Variance monitoring
8. PSI Scores:                  Alternative drift metric
9. Prediction Latency:          Performance monitoring


## Alerting
Alerts are triggered when:

1. Data Drift Detected:             Drift score > 0.3 for 1 minute
2. Critical Data Drift:             Drift score > 0.5 for 30 seconds
3. Prediction Distribution Shift:   Rate drops significantly
4. Model Service Down:              Service unreachable for 1 minute


Alerts are sent to Slack with:

1. Alert name and severity
2. Drift score and threshold
3. Feature name
4. Recommended actions


## Full Cleanup
To do a total cleanup of everything, use the cleanup.sh script:

```sh
chmod +x cleanup.sh
./cleanup.sh
```

# Please LIKE, COMMENT, and SUBSCRIBE !!!
