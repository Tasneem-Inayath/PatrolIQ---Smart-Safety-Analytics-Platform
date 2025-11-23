import streamlit as st

# Page config
st.set_page_config(page_title="PatrolIQ - Smart Safety Analytics", layout="wide")

# Title and intro
st.title("🛡️ PatrolIQ - Smart Safety Analytics Platform")
st.markdown("""
Welcome to **PatrolIQ**, an AI-driven urban safety intelligence platform that analyzes  
**500,000+ Chicago crime records** to help law enforcement make data-driven decisions.
""")

st.write("---")
st.markdown("### 🔍 Navigate to Modules:")

# --------- Buttons Layout ----------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Data Analysis"):
        st.switch_page("pages/data_analysis.py")

    if st.button("📍 Geo Clustering"):
        st.switch_page("pages/geo_clustering.py")

with col2:
    if st.button("⏳ Temporal Clustering"):
        st.switch_page("pages/temporal_clustering.py")
    if st.button("🔽 Dimensionality Reduction"):
        st.switch_page("pages/dimensionality_reduction.py")

with col3:

    if st.button("🚓 Patrol Recommendation"):
        st.switch_page("pages/patrol_recommendation.py")

st.write("---")
st.markdown("""
### 💡 About PatrolIQ  
This platform delivers:

- 📍 Crime Hotspot Identification  
- ⏳ Temporal Crime Patterns  
- 🔽 Dimensionality Reduction (PCA, UMAP)  
- 🧪 MLflow Experiment Tracking  
- 🚓 Patrol Deployment Recommendations
""")
