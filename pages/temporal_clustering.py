import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
import os

st.title("⏳ Temporal Crime Clustering — Model & Analysis Viewer")

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
# For local MLflow server use HTTP URI
# For Streamlit Cloud, switch to file-based tracking
USE_SERVER = False   # change to True if you run mlflow server externally

if USE_SERVER:
    TRACKING_URI = "http://127.0.0.1:5000"   # or your remote MLflow server
else:
    TRACKING_URI = "file:./mlruns"           # repo-bundled runs

EXPERIMENT_NAME = "Temporal Clustering"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# -----------------------------------------------------------
# GET EXPERIMENT
# -----------------------------------------------------------
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    st.error(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
    st.stop()

experiment_id = experiment.experiment_id
st.success(f"Loaded Experiment: {EXPERIMENT_NAME}")

# -----------------------------------------------------------
# LIST ALL RUNS
# -----------------------------------------------------------
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["attributes.start_time DESC"]
)

if not runs:
    st.warning("No runs found in this experiment.")
    st.stop()

run_map = {run.info.run_id: run for run in runs}

# Show run dropdown
run_id = st.selectbox(
    "Select a Run",
    options=list(run_map.keys()),
    format_func=lambda r: f"{run_map[r].data.params.get('algorithm', 'Unknown')} | {r}"
)

run = run_map[run_id]

# -----------------------------------------------------------
# SHOW PARAMETERS
# -----------------------------------------------------------
st.subheader("🧩 Parameters")
st.json(run.data.params if run.data.params else {"info": "No parameters logged."})

# -----------------------------------------------------------
# SHOW METRICS
# -----------------------------------------------------------
st.subheader("📈 Metrics")
st.json(run.data.metrics if run.data.metrics else {"info": "No metrics logged."})

# -----------------------------------------------------------
# ARTIFACT VIEWER
# -----------------------------------------------------------
st.subheader("📂 Temporal Analysis Artifacts")

# Direct file access for Streamlit Cloud
# -----------------------------------------------------------
# ARTIFACT VIEWER — NO OS MODULE, STREAMLIT-CLOUD SAFE
# -----------------------------------------------------------
st.subheader("📂 Temporal Analysis Artifacts")

artifact_dir = "etc"   # your existing folder inside repo

known_artifacts = [
    "day_hour_heatmap.png",
    "hourly_pattern.png",
    "monthly_pattern.png",
    "seasonal_pattern.png",
    "weekday_pattern.png",
]

for art_file in known_artifacts:
    full_path = f"{artifact_dir}/{art_file}"   # simple path

    try:
        st.image(full_path, caption=f"🖼️ {art_file}")
    except:
        st.warning(f"Missing or unreadable: {full_path}")

