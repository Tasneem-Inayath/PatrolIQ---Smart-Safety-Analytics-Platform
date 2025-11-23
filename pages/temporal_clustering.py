import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

st.title("⏳ Temporal Crime Clustering — Model & Analysis Viewer")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Temporal Clustering"

mlflow.set_tracking_uri(TRACKING_URI)

# -----------------------------------------------------------
# GET EXPERIMENT
# -----------------------------------------------------------
client = MlflowClient()
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
if run.data.params:
    st.json(run.data.params)
else:
    st.info("No parameters logged.")

# -----------------------------------------------------------
# SHOW METRICS
# -----------------------------------------------------------
st.subheader("📈 Metrics")
if run.data.metrics:
    st.json(run.data.metrics)
else:
    st.info("No metrics logged.")

# -----------------------------------------------------------
# ARTIFACT VIEWER
# -----------------------------------------------------------
st.subheader("📂 Temporal Analysis Artifacts")

def list_artifacts(path=""):
    files = client.list_artifacts(run_id, path)
    all_files = []
    for f in files:
        if f.is_dir:
            all_files += list_artifacts(f.path)
        else:
            all_files.append(f.path)
    return all_files

artifact_files = list_artifacts()

if not artifact_files:
    st.warning("No artifacts found for this run.")
else:
    artifact = st.selectbox("Choose an Artifact", artifact_files)

    if artifact:
        artifact_bytes = client.download_artifacts(run_id, artifact)

        if artifact.lower().endswith((".png", ".jpg", ".jpeg")):
            st.image(artifact_bytes)
        else:
            st.download_button(
                label="Download Artifact",
                data=artifact_bytes,
                file_name=os.path.basename(artifact),
            )

# -----------------------------------------------------------
# LOAD TEMPORAL MODEL
# -----------------------------------------------------------
st.subheader("🧠 Load Temporal Model")

model_uri = f"runs:/{run_id}/Temporal_KMeans_Model"

try:
    model = mlflow.sklearn.load_model(model_uri)
    st.success("Temporal KMeans model loaded successfully!")
except Exception as e:
    st.warning(f"No model found in this run. ({e})")
