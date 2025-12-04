import streamlit as st
import pandas as pd

st.set_page_config(page_title="PatrolIQ - Smart Crime Analytics", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crimes_2023_to_Present_2025_enhanced.csv")

df = load_data()

# Title + Intro
st.title("🚔 PatrolIQ — Smart Safety Analytics Platform")
st.markdown("### 💡 AI-powered Crime Pattern Intelligence for Public Safety")

st.write("""
Welcome to **PatrolIQ** — an interactive data-driven safety intelligence system.

Use the quick actions below or the sidebar menu to explore crime patterns:

""")

st.success(f"Dataset Loaded Successfully — **{df.shape[0]:,} rows** | **{df.shape[1]} columns**")

# -------------------------------
# 🧭 Navigation Buttons
# -------------------------------
st.markdown("## 🔍 Explore Analysis Modules")

col1, col2 = st.columns(2)

with col1:
    if st.button("📍 Geographic Hotspots"):
        st.switch_page("pages/Geographic_Hotspots.py")
    
    if st.button("📉 Dimensionality Reduction (PCA + UMAP)"):
        st.switch_page("pages/Dimensionality_Reduction_PCA_UMAP.py")

with col2:
    if st.button("⏳ Temporal Crime Patterns"):
        st.switch_page("pages/Temporal_Patterns.py")

    if st.button("📂 Model Registry & Metrics (MLflow)"):
        st.switch_page("pages/Model_Registry_and_Metrics.py")


# Footer
st.markdown("---")
st.write("🔒 *PatrolIQ is an AI-powered research prototype designed for policing strategy and public safety insight.*")
