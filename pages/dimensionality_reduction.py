import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# -----------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------
st.title("🎛️ PatrolIQ — Dimensionality Reduction Viewer (PCA & UMAP)")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "PatrolIQ-PCA"  # For PCA
UMAP_EXPERIMENT_NAME = "UMAP_Crime_Clusters_Registered"  # For UMAP

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

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

        # Load PCA model
        try:
            pca_model = mlflow.sklearn.load_model(f"runs:/{pca_run_id}/pca_model")
            st.success("PCA Model Loaded Successfully!")
        except:
            st.warning("No PCA model found in this run.")

        # Show PCA artifacts (like scree plot)
        st.subheader("📂 PCA Artifacts")
        artifacts = client.list_artifacts(pca_run_id)
        for art in artifacts:
            if art.path.endswith(".png"):
                st.image(client.download_artifacts(pca_run_id, art.path))
            elif art.path.endswith(".csv"):
                df = pd.read_csv(client.download_artifacts(pca_run_id, art.path))
                st.dataframe(df.head())
                st.download_button("Download CSV", data=open(client.download_artifacts(pca_run_id, art.path), "rb"),
                                   file_name=art.path)

# -----------------------------------------------------------
# LOAD UMAP EXPERIMENT
# -----------------------------------------------------------
st.header("🌀 UMAP Viewer")

umap_experiment = client.get_experiment_by_name("PatrolIQ-UMAP")  # Adjust to your registered UMAP experiment name
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

        # Load UMAP pyfunc model
        try:
            umap_model = mlflow.pyfunc.load_model(f"runs:/{umap_run_id}/UMAP_Model")
            st.success("UMAP Model Loaded Successfully!")
        except:
            st.warning("No UMAP model found in this run.")

        # Show UMAP artifacts
        st.subheader("📂 UMAP Artifacts")
        artifacts = client.list_artifacts(umap_run_id)
        for art in artifacts:
            if art.path.endswith(".png"):
                st.image(client.download_artifacts(umap_run_id, art.path))
            elif art.path.endswith(".csv"):
                df = pd.read_csv(client.download_artifacts(umap_run_id, art.path))
                st.dataframe(df.head())
                st.download_button("Download CSV", data=open(client.download_artifacts(umap_run_id, art.path), "rb"),
                                   file_name=art.path)
