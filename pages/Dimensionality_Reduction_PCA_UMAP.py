import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Dimensionality Reduction Analysis", layout="wide")
st.title("📊 Dimensionality Reduction: PCA & UMAP Crime Pattern Discovery")


# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Crimes_2023_to_Present_2025_enhanced.csv").sample(20000, random_state=42)

df = load_data()

# Select numeric features for embedding
features = [
    'Latitude','Longitude','Beat','District','Ward',
    'Community Area','Hour','CrimeSeverity'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------
# Sidebar Category Selector
# -------------------------
color_mode = st.sidebar.selectbox(
    "Color points by:",
    ["Primary Type", "DistrictCluster", "Season"],
    index=0
)


# -------------------------
# Load PCA Model from MLflow
# -------------------------
st.subheader("🧠 PCA (Principal Component Analysis) — MLflow Model Loaded")

pca_model = mlflow.sklearn.load_model("models:/PCA_Model/latest")#type: ignore
pca_result = pca_model.transform(X_scaled)#type: ignore

df["PCA1"] = pca_result[:,0]
df["PCA2"] = pca_result[:,1]
df["PCA3"] = pca_result[:,2]

# Variance Explained
variance = np.round(pca_model.explained_variance_ratio_ * 100, 2)#type: ignore

st.write(f"📌 **Total variance captured by first 3 components:** `{sum(variance[:3])}%`")


# -------------------------
# Scree Plot
# -------------------------
st.subheader("📈 PCA Scree Plot (Information Retained)")

fig_scree, ax_scree = plt.subplots(figsize=(6,4))
ax_scree.plot(range(1, len(variance)+1), variance, marker='o')
ax_scree.set_title("Variance vs PCA Component")
ax_scree.set_xlabel("Component Number")
ax_scree.set_ylabel("Variance Explained (%)")
st.pyplot(fig_scree)


# -------------------------
# PCA 2D Visualization
# -------------------------
st.subheader("🎯 PCA 2D Visualization (Crime Pattern Space)")

fig_2d = px.scatter(
    df.sample(8000),
    x="PCA1", y="PCA2",
    color=color_mode,
    opacity=0.7,
    title="PCA Crime Projection — 2D",
)
st.plotly_chart(fig_2d, use_container_width=True)


# -------------------------
# PCA 3D Visualization
# -------------------------
st.subheader("🌍 Interactive 3D PCA Visualization")

fig_3d = px.scatter_3d(
    df.sample(8000),
    x="PCA1", y="PCA2", z="PCA3",
    color=color_mode,
    opacity=0.7,
    title="PCA Crime Projection — 3D",
)
st.plotly_chart(fig_3d, use_container_width=True)


# -------------------------
# PCA Feature Importance (Loadings)
# -------------------------
st.subheader("🔍 Feature Importance in PCA (Which variables influence patterns?)")

loadings = pca_model.components_.T[:, :3]#type: ignore
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'PC1_Impact': np.abs(loadings[:,0]),
    'PC2_Impact': np.abs(loadings[:,1]),
    'PC3_Impact': np.abs(loadings[:,2]),
})

feature_importance["Total_Impact"] = feature_importance.iloc[:,1:].sum(axis=1)
feature_importance = feature_importance.sort_values("Total_Impact", ascending=False)

fig_feat, ax_feat = plt.subplots(figsize=(8,5))
ax_feat.bar(feature_importance["Feature"], feature_importance["Total_Impact"])
ax_feat.set_title("Top Feature Influence (PCA)")
ax_feat.set_ylabel("Influence Weight")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig_feat)


# -------------------------
# UMAP SECTION
# -------------------------
features_for_umap = [
    'Latitude','Longitude','Hour','Day','Month',
    'CrimeSeverity','DistrictCluster'
]
# -------------------------
# 📌 Load UMAP Model with Exact Training Feature Names
# -------------------------
st.subheader("🔮 UMAP Crime Embedding (Loaded from MLflow)")

import pandas as pd

df = pd.read_csv("Crimes_2023_to_Present_2025_enhanced.csv").sample(20000, random_state=42)
# ---- Extract datetime components ----
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = df['Date'].dt.hour#type: ignore
df['Day'] = df['Date'].dt.dayofweek  #type: ignore     # 0=Mon, 6=Sun
df['Month'] = df['Date'].dt.month#type: ignore
df['IsWeekend'] = df['Day'].apply(lambda x: 1 if x>=5 else 0)
features_for_umap = [
    'Latitude','Longitude','Hour','Day','Month',
    'CrimeSeverity','DistrictCluster'
]

X_umap = df[features_for_umap]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_umap)
import umap

umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)

umap_2d = umap_model.fit_transform(X_scaled) # convert string list to python list



df["UMAP1"] = umap_2d[:,0]#type: ignore
df["UMAP2"] = umap_2d[:,1]#type: ignore

# Plot UMAP
fig_umap = px.scatter(
    df.sample(12000, random_state=42),
    x="UMAP1", y="UMAP2",
    color=color_mode,
    opacity=0.7,
    title="UMAP Embedding — Crime Behavior Similarity"
)

st.plotly_chart(fig_umap, use_container_width=True)


st.success("✨ Dimensionality Reduction Analysis Loaded Successfully")
