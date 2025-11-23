import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Patrol Recommendation", layout="wide")
st.title("🚓 Patrol Recommendation — PatrolIQ")

# ----------------------------------------------------
# MLflow Setup
# ----------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# ----------------------------------------------------
# Load Registered Models
# ----------------------------------------------------
st.subheader("📦 Loading Registered Models")


try:
    geo_model = mlflow.sklearn.load_model( #type: ignore
        "mlartifacts/873737492296709931/models/m-d639a32a33cc4edf947fa9d1a13a77e8/artifacts"
    )
    st.success("Loaded DBSCAN Model from artifacts folder 🎉")
except Exception as e:
    st.error(f"❌ Failed to load DBSCAN_Model: {e}")
    geo_model = None


try:
    temp_model = mlflow.sklearn.load_model("mlartifacts/702219540321182259/models/m-86961cb4b5214fd68795e12265b31ff2/artifacts")#type: ignore
    st.success("Loaded model: Temporal_KMeans_Model")
except Exception as e:
    st.error(f"❌ Failed to load Temporal_KMeans_Model: {e}")
    temp_model = None

# ----------------------------------------------------
# Load Crime Dataset
# ----------------------------------------------------
DATA_PATH = "data/Crimes_2023_to_2025_CLEANED.csv"
try:
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)  # Reduced sample for speed
    st.success("Crime dataset loaded.")
except:
    st.error("❌ Dataset NOT found. Please check path.")
    st.stop()

# ----------------------------------------------------
# Preprocess columns for clustering
# ----------------------------------------------------
if 'Day_of_Week_Num' not in df.columns:
    df['Day_of_Week_Num'] = df['Day_of_Week'].map({
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    })

# ----------------------------------------------------
# Apply Geo Clustering (DBSCAN)
# ----------------------------------------------------
if geo_model:
    st.subheader("🌍 Applying Geo Clustering (DBSCAN)")
    geo_features = df[["Latitude", "Longitude"]]
    df["geo_cluster"] = geo_model.fit_predict(geo_features)  # DBSCAN does not support predict
    st.write(df["geo_cluster"].value_counts())

# ----------------------------------------------------
# Apply Temporal Clustering (KMeans)
# ----------------------------------------------------
if temp_model:
    st.subheader("⏱️ Applying Temporal Clustering (KMeans)")
    temp_features = df[["Hour", "Day_of_Week_Num", "Month"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(temp_features)
    df["temp_cluster"] = temp_model.predict(scaled)
    st.write(df["temp_cluster"].value_counts())

# ----------------------------------------------------
# Compute Patrol Priority Scores
# ----------------------------------------------------
st.subheader("🔥 Computing Patrol Risk Score")

def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

cluster_summary = (
    df.groupby(['geo_cluster', 'temp_cluster'])
      .agg({
          'Latitude': 'mean',
          'Longitude': 'mean',
          'geo_cluster': 'count'
      })
      .rename(columns={'geo_cluster': 'crime_count'})
      .reset_index()
)

cluster_summary['geo_risk'] = norm(cluster_summary['crime_count'])
cluster_summary['temp_risk'] = norm(
    cluster_summary.groupby('temp_cluster')['crime_count'].transform('sum')
)
cluster_summary['final_risk'] = 0.6 * cluster_summary['geo_risk'] + 0.4 * cluster_summary['temp_risk']

top_hotspots = cluster_summary.sort_values('final_risk', ascending=False).head(5)
st.write("### 🚓 Top 5 Patrol Hotspots")
st.dataframe(top_hotspots[['Latitude', 'Longitude', 'final_risk']])

# ----------------------------------------------------
# Map Visualization
# ----------------------------------------------------
st.subheader("🗺️ Patrol Map")
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
for _, row in top_hotspots.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=7,
        popup=f"Risk: {row['final_risk']:.3f}",
        color='red',
        fill=True,
    ).add_to(m)

st_folium(m, width=900, height=500)
