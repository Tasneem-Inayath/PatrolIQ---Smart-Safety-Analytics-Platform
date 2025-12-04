import streamlit as st
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow

# -----------------------------------------------
# PAGE SETUP
# -----------------------------------------------
st.set_page_config(page_title="MLflow Model Dashboard", layout="wide")
st.title("📂 MLflow Model Registry & Experiment Dashboard")

client = MlflowClient()


# -----------------------------------------------
# 1️⃣ LIST REGISTERED MODELS (COMPATIBLE)
# -----------------------------------------------
st.subheader("📌 Registered Models")

try:
    registered_models = client.search_registered_models()
except Exception:
    registered_models = []

if not registered_models:
    st.info("No registered models found in MLflow Registry.")
else:
    model_names = [m.name for m in registered_models]
    selected_model = st.selectbox("Select Model", model_names)

    st.write(f"### 🧠 Selected Model: `{selected_model}`")

    versions = client.get_latest_versions(selected_model)
    version_info = []

    for v in versions:
        run = client.get_run(v.run_id) #type:ignore
        version_info.append({
            "Version": v.version,
            "Stage": v.current_stage,
            "Run ID": v.run_id,
            "Source": v.source,
            "Created": pd.to_datetime(run.info.start_time, unit='ms')
        })

    st.dataframe(pd.DataFrame(version_info))


    # -----------------------------------------------
    # 2️⃣ MODEL RUN DETAILS (PARAMS + METRICS)
    # -----------------------------------------------
    st.subheader("📊 Model Run Metadata")

    selected_version = st.selectbox(
        "Inspect Version",
        [v["Version"] for v in version_info]
    )

    run_id = [x for x in version_info if x["Version"] == selected_version][0]["Run ID"]
    run_data = client.get_run(run_id)

    col1, col2 = st.columns(2)

    with col1:
        st.write("📌 Parameters")
        st.json(run_data.data.params or "No parameters logged")

    with col2:
        st.write("📈 Metrics")
        st.json(run_data.data.metrics or "No metrics logged")



