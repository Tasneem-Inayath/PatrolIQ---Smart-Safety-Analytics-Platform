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
          # PCA Artifacts in etc/
        pca_plot_path = "etc/pca_variance_plot.png"
        
        if os.path.exists(pca_plot_path):
            st.subheader("📊 PCA Variance Plot")
            st.image(pca_plot_path, caption="PCA Variance Explained")
        else:
            st.warning("PCA variance plot not found in /etc folder.")

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
        
        st.header("🌀 UMAP Viewer")
        
        umap_plot_1 = "etc/umap_crime_type.png"
        umap_plot_2 = "etc/umap_day_night.png"
        umap_csv = "etc/umap_embeddings.csv"
        
        # UMAP Image 1
        if os.path.exists(umap_plot_1):
            st.subheader("🔹 UMAP — Crime Type")
            st.image(umap_plot_1, caption="UMAP Clustering by Crime Type")
        else:
            st.warning("UMAP crime type image not found.")
        
        # UMAP Image 2
        if os.path.exists(umap_plot_2):
            st.subheader("🔹 UMAP — Day vs Night Pattern")
            st.image(umap_plot_2, caption="UMAP Day–Night Embeddings")
        else:
            st.warning("UMAP day/night image not found.")
        
        # UMAP Embeddings CSV
        if os.path.exists(umap_csv):
            st.subheader("📄 UMAP Embeddings CSV")
            df = pd.read_csv(umap_csv)
            st.dataframe(df.head())
        
            with open(umap_csv, "rb") as f:
                st.download_button(
                    label="Download UMAP Embeddings CSV",
                    data=f,
                    file_name="umap_embeddings.csv"
                )
        else:
            st.warning("UMAP embeddings CSV not found.")
