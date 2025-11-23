import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

# -----------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------
st.title("🎛️ PatrolIQ — Dimensionality Reduction Viewer (PCA & UMAP)")

EXPERIMENT_NAME = "PatrolIQ-PCA"  # For PCA
UMAP_EXPERIMENT_NAME = "UMAP_Crime_Clusters_Registered"  # For UMAP

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# -----------------------------------------------------------
# Helper: get artifact folder path
# -----------------------------------------------------------
def get_artifact_folder(run_id):
    # MLflow stores artifacts under mlartifacts/<experiment_id>/<run_id>/artifacts
    return os.path.join("mlartifacts", run_id, "artifacts")

# -----------------------------------------------------------
# LOAD PCA EXPERIMENT
# -----------------------------------------------------------
st.header("📐 PCA Viewer")

pca_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if pca_experiment is None:
    st.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
else:
    st.success(f"Loaded PCA Experiment: {EXPERIMENT_NAME}")
    pca_runs = client.search_runs([pca_experiment.experiment_id], order_by=["attributes.start_time DESC"])
    if not pca_runs:
        st.warning("No PCA runs found.")
    else:
        pca_run_map = {run.info.run_id: run for run in pca_runs}
        pca_run_id = st.selectbox(
            "Select PCA Run",
            options=list(pca_run_map.keys()),
            format_func=lambda r: f"{pca_run_map[r].data.params.get('algorithm', 'Unknown')} | {r}"
        )
        pca_run = pca_run_map[pca_run_id]
        st.subheader("📌 PCA Parameters")
        st.json(pca_run.data.params)
        st.subheader("📊 PCA Metrics")
        st.json(pca_run.data.metrics)


        # Show PCA artifacts
        # Show PCA artifacts
        st.subheader("📂 PCA Artifacts")

        # Use the artifacts directory, not a single file
        artifacts_dir = r"mlartifacts\150500204737805743\3ee91f4ff41d42259a433095768e61e0\artifacts"

        if os.path.exists(artifacts_dir):
            for art in os.listdir(artifacts_dir):
                art_file = os.path.join(artifacts_dir, art)
                if art.endswith(".png"):
                    st.image(art_file, caption=art)
                elif art.endswith(".csv"):
                    df = pd.read_csv(art_file)
                    st.dataframe(df.head())
                    with open(art_file, "rb") as f:
                        st.download_button("Download CSV", data=f, file_name=art)
        else:
            st.warning("No artifacts folder found for this run.")

# -----------------------------------------------------------
# LOAD UMAP EXPERIMENT
# -----------------------------------------------------------
st.header("🌀 UMAP Viewer")

umap_experiment = client.get_experiment_by_name("PatrolIQ-UMAP")
if umap_experiment is None:
    st.error(f"Experiment 'PatrolIQ-UMAP' not found.")
else:
    st.success("Loaded UMAP Experiment: PatrolIQ-UMAP")
    umap_runs = client.search_runs([umap_experiment.experiment_id], order_by=["attributes.start_time DESC"])
    if not umap_runs:
        st.warning("No UMAP runs found.")
    else:
        umap_run_map = {run.info.run_id: run for run in umap_runs}
        umap_run_id = st.selectbox(
            "Select UMAP Run",
            options=list(umap_run_map.keys()),
            format_func=lambda r: f"{umap_run_map[r].data.params.get('algorithm', 'Unknown')} | {r}"
        )
        umap_run = umap_run_map[umap_run_id]
        st.subheader("📌 UMAP Parameters")
        st.json(umap_run.data.params)
        st.subheader("📊 UMAP Metrics")
        st.json(umap_run.data.metrics)


        # Show UMAP artifacts

        # Show UMAP artifacts
        st.subheader("📂 UMAP Artifacts")

        # Use the exact folder path from your image
        umap_artifacts_dir = r"mlartifacts\830657570369720766\64cd0281de57407f8a3566046961f2c3\artifacts"

        if os.path.exists(umap_artifacts_dir):
            for art in os.listdir(umap_artifacts_dir):
                art_file = os.path.join(umap_artifacts_dir, art)
                if art.endswith(".png"):
                    st.image(art_file, caption=art)
                elif art.endswith(".csv"):
                    df = pd.read_csv(art_file)
                    st.dataframe(df.head())
                    with open(art_file, "rb") as f:
                        st.download_button("Download CSV", data=f, file_name=art)
        else:
            st.warning("No artifacts found in the specified UMAP folder.")