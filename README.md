# 🚛 DPF Soot Load Prediction & Maintenance Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-ready end-to-end Data Science + MLOps pipeline for predictive maintenance of Diesel Particulate Filters (DPF) in commercial vehicle fleets.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Domain Context](#-domain-context)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Pipeline Execution](#-pipeline-execution)
- [API Documentation](#-api-documentation)
- [Real-time Mode](#-real-time-mode-with-redis)
- [Testing & Validation](#-testing--validation)
- [Docker Deployment](#-docker-deployment)
- [Monitoring & Observability](#-monitoring--observability)
- [Production Considerations](#-production-considerations)
- [Technologies Used](#-technologies-used)

---

## 🎯 Overview

This project implements a **complete predictive maintenance system** for monitoring and predicting **DPF (Diesel Particulate Filter) soot load** in medium and heavy-duty diesel commercial vehicles. The system predicts soot accumulation levels and recommends proactive maintenance actions to prevent:

- ⚠️ Engine derate events
- 🔥 Forced regenerations
- ⛽ Increased fuel consumption
- 🚫 Unplanned downtime
- 💰 Component damage

### Problem Formulation

**Primary Task:** Regression model to predict `soot_load_pct` (0-100%)  
**Secondary Task:** Rule-based recommendation engine to convert predictions into actionable maintenance decisions

### Maintenance Recommendations

| Action | Threshold | Description |
|--------|-----------|-------------|
| ✅ **OK** | < 60% | Normal operation |
| 👀 **MONITOR** | 60-70% | Increased monitoring frequency |
| 🌡️ **PASSIVE_REGEN_OPPORTUNITY** | 70-80% | Schedule highway driving for passive regeneration |
| 🔧 **ACTIVE_REGEN** | 80-90% | Trigger active regeneration cycle |
| 🔍 **INSPECTION** | > 90% | Immediate inspection required |

---

## ✨ Key Features

### Data Engineering
- ✅ **Synthetic Multi-Table Dataset Generation** (Telemetry, Trips, Maintenance)
- ✅ **Time-Aware Joins** using `merge_asof` for temporal data alignment
- ✅ **Advanced Feature Engineering** (rolling statistics, trend detection, driving mode analysis)
- ✅ **Data Quality Checks** and versioning strategy

### Machine Learning
- ✅ **LightGBM Regression Model** with optimized performance
- ✅ **Optuna Hyperparameter Tuning** for automated optimization
- ✅ **Prediction Intervals** (confidence intervals) for uncertainty quantification
- ✅ **Model Evaluation** with business-aware metrics

### MLOps & Production
- ✅ **FastAPI REST API** with comprehensive endpoints
- ✅ **Batch & Real-time Inference** capabilities
- ✅ **Redis-based Feature Store** for streaming data
- ✅ **Docker Containerization** for easy deployment
- ✅ **Monitoring & Logging** with drift detection
- ✅ **Comprehensive Test Suite** including edge cases

---

## 🔧 Domain Context

### What is a DPF?

A **Diesel Particulate Filter (DPF)** is a critical component in modern diesel vehicles that traps harmful soot particles from exhaust gases. Over time, soot accumulates within the filter, increasing backpressure and reducing engine performance.

### Regeneration Types

| Type | Trigger | Description |
|------|---------|-------------|
| **Passive Regeneration** | Natural | Occurs during sustained high-temperature driving (e.g., highway cruising at 70+ mph) |
| **Active Regeneration** | ECU-initiated | Engine control unit injects extra fuel to raise exhaust temperature above 600°C |
| **Forced Regeneration** | Manual/Service | Performed during maintenance using diagnostic equipment |

### Why Predictive Maintenance?

Traditional reactive maintenance leads to:
- 🚨 Unexpected breakdowns and costly towing
- ⏱️ Extended vehicle downtime
- 💸 Emergency repair premiums
- 📉 Reduced fleet productivity

**Our solution provides:**
- 📊 Proactive maintenance scheduling
- 💰 Cost optimization (reduce unnecessary interventions)
- 🎯 Precision timing (intervene before critical failure)
- 📈 Extended component lifespan

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Telemetry   │  │    Trips     │  │ Maintenance  │         │
│  │  (Sensors)   │  │ Characteristics│ │   Records    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DATA ENGINEERING LAYER                          │
│  • Time-aware joins (merge_asof)                                │
│  • Data quality validation                                       │
│  • Feature engineering (rolling stats, trends, modes)           │
│  • Data versioning                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODELING LAYER                                │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │  LightGBM Model  │  ◄────  │ Optuna HPT      │              │
│  │  (Regression)    │         │ (AutoML)        │              │
│  └──────────────────┘         └─────────────────┘              │
│           ↓                                                      │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │ Prediction       │         │ Recommendation  │              │
│  │ Intervals (CI)   │         │ Engine (Rules)  │              │
│  └──────────────────┘         └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT LAYER (MLOps)                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              FastAPI REST API                           │    │
│  │  • /predict/soot-load  • /predict/batch                │    │
│  │  • /predict/from-raw   • /ingest/telemetry             │    │
│  │  • /model/info         • /health                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │    Redis     │  │  Monitoring  │  │    Docker    │         │
│  │Feature Store │  │ & Drift Det. │  │ Container    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
dpfsootprediction/
│
├── 📂 src/                           # Source code
│   ├── 📂 api/                       # FastAPI application
│   │   ├── main.py                   # API endpoints
│   │   └── schemas.py                # Pydantic data models
│   │
│   ├── 📂 data_generation/           # Synthetic data generators
│   │   ├── generate_telemetry.py
│   │   ├── generate_trips.py
│   │   └── generate_maintenance.py
│   │
│   ├── 📂 data_engineering/          # Data pipeline
│   │   ├── join.py                   # Time-aware joins
│   │   └── validation.py             # Data quality checks
│   │
│   ├── 📂 features/                  # Feature engineering
│   │   ├── rolling_features.py
│   │   ├── trend_features.py
│   │   └── driving_mode_features.py
│   │
│   ├── 📂 models/                    # ML models
│   │   ├── train_regression.py       # Baseline training
│   │   ├── hpt_optuna.py             # Hyperparameter tuning
│   │   └── uncertainty.py            # Confidence intervals
│   │
│   ├── 📂 recommendation/            # Business logic
│   │   └── recommender.py            # Action recommendation rules
│   │
│   ├── 📂 monitoring/                # Observability
│   │   ├── logger.py                 # Prediction logging
│   │   └── drift.py                  # Data drift detection
│   │
│   ├── 📂 feature_store/             # Real-time features
│   │   └── redis_store.py            # Redis integration
│   │
│   └── 📂 realtime/                  # Streaming features
│       └── feature_builder.py        # Real-time feature generation
│
├── 📂 tests/                         # Test suite
│   ├── test_features.py              # Unit tests
│   ├── test_model.py                 # Model tests
│   └── 📂 edge_cases/
│       └── test_edge_cases.py        # Edge case scenarios
│
├── 📂 scripts/                       # Utility scripts
│   ├── final_api_eval.py             # API validation
│   └── compare_local_vs_api.py       # Sanity checks
│
├── 📂 data/                          # Generated datasets
│   ├── telemetry.parquet
│   ├── trips.parquet
│   ├── maintenance.parquet
│   └── ml_base.parquet
│
├── 📂 models/                        # Model artifacts
│   ├── soot_regressor_lgbm.joblib
│   ├── soot_regressor_optuna.joblib
│   ├── feature_list.joblib
│   ├── prediction_interval.joblib
│   └── baseline_feature_stats.parquet
│
├── 📂 logs/                          # Application logs
│   └── predictions.jsonl
│
├── 🐳 Dockerfile                     # Container definition
├── 📋 requirements.txt               # Python dependencies
├── 🚀 run_*.py                       # Pipeline orchestration scripts
└── 📖 README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Redis (for real-time mode)
- Docker (optional, for containerized deployment)

### 1. Clone Repository

```bash
git clone https://github.com/darshanganatra01/dpfsootprediction.git
cd dpfsootprediction
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import lightgbm, fastapi, optuna, redis; print('✅ All dependencies installed successfully!')"
```

---

## 🔄 Pipeline Execution

Execute the complete pipeline in sequence:

### Step 1: Generate Synthetic Data

```bash
python run_generate.py
```

**Output:**
- `data/telemetry.parquet` (minute-level sensor readings)
- `data/trips.parquet` (trip-level aggregates)
- `data/maintenance.parquet` (regeneration & inspection events)

**What it does:** Creates realistic synthetic datasets with:
- 50 vehicles
- 30 days of operation
- Correlated sensor readings (exhaust temp ↑ → soot load ↓)
- Realistic noise and edge cases

---

### Step 2: Join Datasets

```bash
python run_join.py
```

**Output:**
- `data/ml_base.parquet` (unified ML-ready table)

**What it does:**
- Time-aware joins using `merge_asof`
- Aligns telemetry with trip context
- Incorporates maintenance history
- Handles temporal misalignments

---

### Step 3: Feature Engineering

```bash
python run_features.py
```

**Output:**
- `data/ml_features.parquet` (engineered features)

**What it does:**
- **Rolling statistics**: 5/15/30-min windows for exhaust temp, differential pressure
- **Trend detection**: Temperature gradients indicating regeneration opportunities
- **Driving mode features**: Idle time %, city vs highway distribution
- **Temporal features**: Hours since last regeneration
- **Interaction terms**: Load × RPM, temp differential × flow rate

**Example Features:**
```python
- exhaust_temp_rolling_15min_mean
- diff_pressure_trend_30min
- idle_time_pct_last_hour
- hours_since_last_regen
- high_load_duration_minutes
```

---

### Step 4: Train Baseline Model

```bash
python run_train.py
```

**Output:**
```
models/
  ├── soot_regressor_lgbm.joblib          # Trained model
  ├── feature_list.joblib                  # Feature names
  ├── baseline_feature_stats.parquet       # Stats for drift detection
  └── prediction_interval.joblib           # Uncertainty estimator
```

**Performance Metrics:**
- MAE: ~2.5% (mean absolute error in soot load prediction)
- RMSE: ~3.8%
- R²: ~0.92

---

### Step 5: Hyperparameter Tuning (Optional but Recommended)

```bash
python run_hpt.py
```

**Output:**
```
models/
  ├── soot_regressor_optuna.joblib         # Optimized model
  ├── soot_regressor_optuna_feature_list.joblib
  └── soot_regressor_optuna_best_params.joblib
```

**What it does:**
- Runs 100 Optuna trials
- Optimizes learning rate, max depth, num leaves, regularization
- Typically improves MAE by 0.3-0.5%

**Tuned Hyperparameters Example:**
```json
{
  "n_estimators": 250,
  "learning_rate": 0.05,
  "max_depth": 8,
  "num_leaves": 45,
  "min_child_samples": 30,
  "reg_alpha": 0.1,
  "reg_lambda": 0.5
}
```

---

## 🌐 API Documentation

### Start the API Server

```bash
python run_api.py
```

Server starts at: `http://localhost:8000`  
Swagger UI (interactive docs): `http://localhost:8000/docs`

---

### 📍 API Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "timestamp": "2026-01-16T10:30:00Z"
}
```

---

#### 2. Model Information

```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "LightGBM Regressor",
  "version": "v1.2.3",
  "training_date": "2026-01-15",
  "features_count": 42,
  "performance_metrics": {
    "mae": 2.45,
    "rmse": 3.72,
    "r2": 0.923
  },
  "feature_importance_top5": [
    "exhaust_temp_rolling_15min_mean",
    "hours_since_last_regen",
    "diff_pressure_trend_30min",
    "high_load_duration_minutes",
    "idle_time_pct_last_hour"
  ]
}
```

---

#### 3. Predict Soot Load (Engineered Features)

```http
POST /predict/soot-load
```

**Request Body:**
```json
{
  "vehicle_id": "V001",
  "features": {
    "exhaust_temp_rolling_15min_mean": 385.5,
    "diff_pressure_kpa": 14.2,
    "hours_since_last_regen": 48.5,
    "idle_time_pct_last_hour": 0.15,
    "high_load_duration_minutes": 25,
    "engine_load_pct": 62,
    "vehicle_speed_kmh": 75
  }
}
```

**Response:**
```json
{
  "vehicle_id": "V001",
  "predicted_soot_load_pct": 78.5,
  "confidence_interval": {
    "lower": 72.1,
    "upper": 84.9,
    "confidence_level": 0.95
  },
  "recommendation": "PASSIVE_REGEN_OPPORTUNITY",
  "recommendation_details": {
    "action": "Schedule highway driving for passive regeneration",
    "urgency": "medium",
    "estimated_time_to_critical": "8-12 hours"
  },
  "model_version": "v1.2.3",
  "prediction_timestamp": "2026-01-16T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict/soot-load" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "V001",
    "features": {
      "exhaust_temp_rolling_15min_mean": 385.5,
      "diff_pressure_kpa": 14.2,
      "hours_since_last_regen": 48.5
    }
  }'
```

---

#### 4. Batch Prediction

```http
POST /predict/batch
```

**Request Body:**
```json
{
  "predictions": [
    {
      "vehicle_id": "V001",
      "features": { /* features */ }
    },
    {
      "vehicle_id": "V002",
      "features": { /* features */ }
    }
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_20260116_103000",
  "total_predictions": 2,
  "predictions": [
    {
      "vehicle_id": "V001",
      "predicted_soot_load_pct": 78.5,
      "recommendation": "PASSIVE_REGEN_OPPORTUNITY"
    },
    {
      "vehicle_id": "V002",
      "predicted_soot_load_pct": 45.2,
      "recommendation": "OK"
    }
  ],
  "processing_time_ms": 124
}
```

---

## ⚡ Real-time Mode with Redis

For streaming telemetry scenarios where you want to predict from the last N minutes of raw sensor data.

### 1. Install & Start Redis

**Linux/WSL:**
```bash
sudo apt-get update
sudo apt-get install -y redis-server
sudo service redis-server start
redis-cli ping  # Should return: PONG
```

**macOS:**
```bash
brew install redis
brew services start redis
redis-cli ping  # Should return: PONG
```

**Docker:**
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

---

### 2. Install Redis Python Client

```bash
pip install redis
```

---

### 3. Real-time Endpoints

#### Ingest Raw Telemetry

```http
POST /ingest/telemetry
```

**Request:**
```json
{
  "vehicle_id": "V001",
  "record": {
    "timestamp": "2026-01-16T10:30:00",
    "vehicle_speed_kmh": 70,
    "engine_load_pct": 55,
    "engine_rpm": 1700,
    "ambient_temp_c": 25,
    "exhaust_temp_pre_dpf_c": 390,
    "exhaust_temp_post_dpf_c": 370,
    "exhaust_flow_rate": 150,
    "differential_pressure_kpa": 12
  }
}
```

**Response:**
```json
{
  "status": "success",
  "vehicle_id": "V001",
  "records_stored": 1,
  "total_records_in_buffer": 45
}
```

---

#### Predict from Raw Telemetry Buffer

```http
POST /predict/from-raw
```

**Request:**
```json
{
  "vehicle_id": "V001",
  "last_n": 60
}
```

**Response:**
```json
{
  "vehicle_id": "V001",
  "predicted_soot_load_pct": 76.3,
  "confidence_interval": {
    "lower": 70.5,
    "upper": 82.1
  },
  "recommendation": "PASSIVE_REGEN_OPPORTUNITY",
  "feature_summary": {
    "records_used": 60,
    "time_window_minutes": 60,
    "avg_exhaust_temp": 385.2,
    "avg_diff_pressure": 13.8
  }
}
```

---

### 4. Simulating Real-time Ingestion

**PowerShell Script (Windows):**
```powershell
# Ingest 60 minutes of telemetry
for ($i=0; $i -lt 60; $i++) {
  $min = "{0:D2}" -f $i
  $body = @{
    vehicle_id="V001"
    record=@{
      timestamp="2026-01-16T08:$min:00"
      vehicle_speed_kmh=70
      engine_load_pct=55
      engine_rpm=1700
      ambient_temp_c=25
      exhaust_temp_pre_dpf_c=390
      exhaust_temp_post_dpf_c=370
      exhaust_flow_rate=150
      differential_pressure_kpa=12
    }
  } | ConvertTo-Json -Depth 10

  Invoke-RestMethod `
    -Uri "http://localhost:8000/ingest/telemetry" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | Out-Null
}
Write-Host "✅ Ingested 60 records"

# Predict from last 60 minutes
$body = @{ vehicle_id="V001"; last_n=60 } | ConvertTo-Json
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict/from-raw" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

**Bash Script (Linux/macOS):**
```bash
#!/bin/bash
# Ingest 60 records
for i in {0..59}; do
  curl -X POST "http://localhost:8000/ingest/telemetry" \
    -H "Content-Type: application/json" \
    -d "{
      \"vehicle_id\": \"V001\",
      \"record\": {
        \"timestamp\": \"2026-01-16T08:$(printf %02d $i):00\",
        \"vehicle_speed_kmh\": 70,
        \"engine_load_pct\": 55,
        \"engine_rpm\": 1700,
        \"exhaust_temp_pre_dpf_c\": 390,
        \"differential_pressure_kpa\": 12
      }
    }" > /dev/null 2>&1
done
echo "✅ Ingested 60 records"

# Predict
curl -X POST "http://localhost:8000/predict/from-raw" \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id": "V001", "last_n": 60}'
```

---

## 🧪 Testing & Validation

### Run Complete Test Suite

```bash
pytest -v
```

**Test Coverage:**
- ✅ Feature engineering functions
- ✅ Model prediction logic
- ✅ API endpoints
- ✅ Edge cases (new DPF, post-regen, sensor failures)
- ✅ Data quality validation

---

### Edge Case Tests

```bash
pytest tests/edge_cases/test_edge_cases.py -v
```

**Scenarios Covered:**
1. **Brand New DPF** (soot load = 0%)
2. **Immediately Post-Regeneration** (soot load < 5%)
3. **Missing Differential Pressure Sensor**
4. **Out-of-Range Temperature Values** (faulty sensor)
5. **Stale/Delayed Data** (timestamp > 10 minutes old)
6. **Extreme Driving Conditions** (100% idle vs 100% highway)

---

### End-to-End Validation

**1. Verify API matches local predictions:**
```bash
python scripts/compare_local_vs_api.py
```

**Expected Output:**
```
✅ Local prediction: 78.45%
✅ API prediction: 78.47%
✅ Absolute difference: 0.02% (PASS)
```

**2. Evaluate deployed API performance:**
```bash
python scripts/final_api_eval.py
```

**Expected Output:**
```
Testing 1000 samples via API...
✅ MAE: 2.48%
✅ RMSE: 3.75%
✅ R²: 0.921
✅ Avg response time: 45ms
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t dpf-soot-prediction:latest .
```

**Image Details:**
- Base: `python:3.9-slim`
- Size: ~450MB
- Includes all dependencies and model artifacts

---

### Run Container

```bash
docker run --rm -p 8000:8000 dpf-soot-prediction:latest
```

**With Redis (Docker Compose):**
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
```

```bash
docker-compose up
```

---

### Verify Deployment

```bash
curl http://localhost:8000/health
```

Access Swagger UI: `http://localhost:8000/docs`

---

## 📊 Monitoring & Observability

### Prediction Logging

All predictions are logged to `logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-01-16T10:30:00Z",
  "vehicle_id": "V001",
  "predicted_soot_load": 78.5,
  "confidence_interval": [72.1, 84.9],
  "recommendation": "PASSIVE_REGEN_OPPORTUNITY",
  "features": { /* feature values */ },
  "model_version": "v1.2.3"
}
```
