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
mlflow.set_tracking_uri("http://127.0.0.1:5000")
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
with st.expander("📘 Run Details"):
    st.write("### Parameters")
    st.json(selected_run.data.params)

    st.write("### Metrics")
    st.json(selected_run.data.metrics)

    st.write("### Artifacts List")
    artifacts = client.list_artifacts(run_id)
    st.write(artifacts)

    # ---- Display artifacts correctly ----
    st.write("### Artifact Previews")

    for art in artifacts:
        art_path = art.path

        try:
            local_file = client.download_artifacts(run_id, art_path)

            if art_path.lower().endswith(".png"):
                st.image(local_file, caption=f"Image: {art_path}")

            elif art_path.lower().endswith(".csv"):
                df_art = pd.read_csv(local_file)
                st.write(f"📄 CSV Preview — {art_path}")
                st.dataframe(df_art)

            else:
                st.write(f"📁 Downloaded: {local_file}")

        except Exception as e:
            st.error(f"Error loading artifact {art_path}: {e}")
