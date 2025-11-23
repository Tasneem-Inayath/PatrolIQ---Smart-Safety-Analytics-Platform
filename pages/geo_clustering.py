import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------------
st.set_page_config(page_title="Geo Clustering", layout="wide")
st.title("🌍 Geo Clustering – PatrolIQ")
st.write("This page loads all clustering runs logged inside the **PatrolIQ-Clustering** experiment.")

# -----------------------------------------------------------
# CONNECT TO MLflow
# -----------------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
EXPERIMENT_NAME = "PatrolIQ-Clustering"
client = MlflowClient()

# -----------------------------------------------------------
# EXPERIMENT + RUNS
# -----------------------------------------------------------
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    st.error("Experiment not found! Please check the experiment name.")
    st.stop()

runs = client.search_runs(experiment_ids=[exp.experiment_id])

if len(runs) == 0:
    st.warning("No runs found in this experiment!")
    st.stop()

# -----------------------------------------------------------
# RUN SELECTION UI
# -----------------------------------------------------------
run_display_names = []
run_map = {}

for r in runs:
    name = f"{r.info.run_id} | Algo: {r.data.params.get('algorithm', 'Unknown')}"
    run_display_names.append(name)
    run_map[name] = r

selected_run_name = st.selectbox("Select a model run:", run_display_names)
selected_run = run_map[selected_run_name]
run_id = selected_run.info.run_id

st.success(f"Loaded Run ID: {run_id}")


# -----------------------------------------------------------
# SHOW METRICS + PARAMETERS + ARTIFACTS
# -----------------------------------------------------------
# -----------------------------------------------------------
# SHOW METRICS + PARAMETERS + ARTIFACTS
# -----------------------------------------------------------
with st.expander("📘 Run Details"):
    st.write("### Parameters")
    st.json(selected_run.data.params)

    st.write("### Metrics")
    st.json(selected_run.data.metrics)

    st.write("### Artifact Previews")

    # Manually list known artifact files for each run
    known_artifacts = {
        "8eda283468304548866781378f5627c4": "dbscan_clusters.png",
        "93c52cd20d7f4289884d24e19422804b": "hier_clusters.png",
        "4845dffa32d44935ab51f7dc01240f04": "kmeans_clusters.png"
    }

    run_artifact_file = known_artifacts.get(run_id)

    if run_artifact_file:
        artifact_path = f"mlartifacts/873737492296709931/{run_id}/artifacts/{run_artifact_file}"
        st.image(artifact_path, caption=f"🖼️ {run_artifact_file}")
    else:
        st.warning("No known artifact file mapped for this run.")
