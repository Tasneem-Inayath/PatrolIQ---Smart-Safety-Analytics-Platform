import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import mlflow.sklearn
import folium
from streamlit_folium import st_folium

# --------------------------------------
# PAGE SETUP
# --------------------------------------
st.set_page_config(page_title="Crime Geo Intelligence System", layout="wide")
st.title("🚓 Crime Geographic Hotspot Intelligence")

# --------------------------------------
# LOAD DATA
# --------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Crimes_2023_to_Present_2025_enhanced.csv").sample(20000, random_state=42)

df = load_data()

# --------------------------------------
# SIDEBAR
# --------------------------------------
st.sidebar.header("⚙ Model Selection")

algo = st.sidebar.selectbox(
    "Select Analysis Mode",
    [
        "KMeans (Preview)",
        "DBSCAN (Preview)",
        "Hierarchical (Preview)",
        "⭐ Final DBSCAN Patrol Assignment"
    ],
    index=0  # Default = KMeans
)

# --------------------------------------
# SCALING GEO FEATURES (Lat/Long)
# --------------------------------------
coords = df[['Latitude', 'Longitude']]
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)


# =============================================================
# ⭐ FINAL MODEL: DBSCAN + TABLE + FOLIUM HOTSPOT MAP
# =============================================================
if algo == "⭐ Final DBSCAN Patrol Assignment":

    st.subheader("🚨 Final Crime Hotspot Deployment (DBSCAN Model from MLflow)")

    # Load DBSCAN model from MLflow
    model = mlflow.sklearn.load_model("models:/DBSCAN_Model/latest") #type:ignore 

    # Predict clusters
    labels = model.fit_predict(coords_scaled) #type:ignore 
    df["PatrolCluster"] = labels

    # Remove noise
    df_clean = df[df["PatrolCluster"] != -1]

    # Compute cluster-level stats
    cluster_stats = (
        df_clean.groupby("PatrolCluster")
        .agg(
            CrimeCount=("Primary Type", "count"),
            TopCrime=("Primary Type", lambda x: x.value_counts().idxmax()),
            District=("District", lambda x: x.value_counts().idxmax())
        )
        .sort_values("CrimeCount", ascending=False)  # highest crime first
        .head(5)
    )

    cluster_stats["Rank"] = range(1, len(cluster_stats) + 1)
    cluster_stats = cluster_stats[["Rank", "CrimeCount", "TopCrime", "District"]]

    st.write("📍 **Top 5 Most Critical Patrol Zones (Cluster-Level)**")
    st.table(cluster_stats)

    # -------------------
    # 🌍 Folium Hotspot Map
    # -------------------
    st.subheader("🗺 Visual Hotspot Map (Top 5 Patrol Zones)")

    center = [df_clean["Latitude"].mean(), df_clean["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=11)

    colors = ["red", "purple", "orange", "blue", "green"]

    # Draw only 5 patrol zones as circles
    for i, cl in enumerate(cluster_stats.index.tolist()):

        cluster_points = df_clean[df_clean["PatrolCluster"] == cl]

        lat_center = cluster_points["Latitude"].mean()
        lon_center = cluster_points["Longitude"].mean()

        # Estimate crime zone area (radius from spread)
        spread = (
            ((cluster_points["Latitude"] - lat_center) ** 2 +
             (cluster_points["Longitude"] - lon_center) ** 2).mean() ** 0.5
        )

        radius = max(150, spread * 120000)  # ensure visible minimum radius

        folium.Circle(
            location=[lat_center, lon_center],
            radius=radius,
            color=colors[i % len(colors)],
            fill=True,
            fill_opacity=0.45,
            popup=(
                f"<b>Patrol Zone #{cluster_stats.iloc[i]['Rank']}</b><br>"
                f"Crime Count: {cluster_stats.iloc[i]['CrimeCount']}<br>"
                f"Top Crime: {cluster_stats.iloc[i]['TopCrime']}<br>"
                f"District: {cluster_stats.iloc[i]['District']}"
            ),
        ).add_to(m)

    st_folium(m, width=900, height=550)

    st.success("✔ Deployment Map and Patrol Table Generated Successfully.")
    def generate_patrol_briefing(df_stats):
        briefing = "🛡 **Operational Patrol Deployment Briefing**\n\n"
        briefing += "Based on the crime clustering analysis using the DBSCAN model, the following districts show the highest concentration of serious or repeating criminal activity. Patrol units are advised to prioritize these locations in order of urgency:\n\n"

        for _, row in df_stats.iterrows():
            zone = row['Rank']
            crime_count = row['CrimeCount']
            top_crime = row['TopCrime'].title()
            district = row['District']

            briefing += (
                f"🔹 **Patrol Zone {zone} — District {district}**\n"
                f"   • Estimated incidents: **{crime_count} cases**\n"
                f"   • Most common crime type: **{top_crime}**\n"
                f"   • Action: Deploy additional patrol units, increase visibility, and monitor peak activity times.\n\n"
            )

        briefing += (
            "---\n"
            "📌 *Recommendation:* Use mobile patrols during low traffic hours and fixed-point surveillance during peak crime hours.\n"
            "📌 *Note:* Continue monitoring evolving hotspots as DBSCAN identifies natural cluster shifts over time.\n"
            
        )

        return briefing
    st.subheader("📢 Automated Patrol Strategy Recommendation")

    instruction_text = generate_patrol_briefing(cluster_stats)
    st.markdown(instruction_text)

    st.stop()


# =============================================================
# 🔍 MODEL PREVIEW SECTION (NOT ACTIONABLE)
# =============================================================

# ⭐ 1) KMEANS Preview
if algo == "KMeans (Preview)":
    st.subheader("📌 KMeans Preview (Testing Only)")
    k = st.sidebar.slider("Choose K", 3, 15, 8)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(coords_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["Longitude"], df["Latitude"], c=labels, s=5, alpha=0.6)
    ax.set_title(f"KMeans Clustering Preview (K={k})")
    st.pyplot(fig)

# ⭐ 2) DBSCAN Preview
elif algo == "DBSCAN (Preview)":
    st.subheader("📌 DBSCAN Preview (Testing Only)")
    eps = st.sidebar.slider("EPS", 0.01, 0.20, 0.05)
    mins = st.sidebar.slider("Min Samples", 5, 150, 60)

    model = DBSCAN(eps=eps, min_samples=mins)
    labels = model.fit_predict(coords_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["Longitude"], df["Latitude"], c=labels, s=5, alpha=0.6)
    ax.set_title(f"DBSCAN Clustering Preview (eps={eps}, min_samples={mins})")
    st.pyplot(fig)

# ⭐ 3) Hierarchical Preview
elif algo == "Hierarchical (Preview)":
    st.subheader("📌 Hierarchical Clustering Preview (Testing Only)")
    st.write("(Using only 3000 sample points for speed)")

    sample = df.sample(3000, random_state=42)
    scaled_sample = scaler.fit_transform(sample[['Latitude', 'Longitude']])

    # ---- Dendrogram ----
    linked = linkage(scaled_sample, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode="level", p=20)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Sample Index (truncated)")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    # ---- Cluster Formation Scatter (like KMeans/DBSCAN) ----
    st.subheader("📌 Hierarchical Cluster Formation (Geo Scatter)")

    # Choose how many clusters to cut the tree into
    n_clusters = st.sidebar.slider("Hierarchical Clusters (for preview)", 3, 12, 6)

    cluster_labels = fcluster(linked, t=n_clusters, criterion="maxclust")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(sample["Longitude"], sample["Latitude"], c=cluster_labels, s=5, alpha=0.6)
    ax2.set_title(f"Hierarchical Clustering Geo Preview (Clusters={n_clusters})")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    st.pyplot(fig2)
