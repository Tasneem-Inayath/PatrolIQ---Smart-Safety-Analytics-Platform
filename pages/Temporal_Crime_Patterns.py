import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

# --------------------------------------
# PAGE CONFIGURATION
# --------------------------------------
st.set_page_config(page_title="Temporal Crime Pattern Intelligence", layout="wide")
st.title("⏱ Crime Temporal Pattern Intelligence Dashboard")


# --------------------------------------
# LOAD DATA
# --------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Crimes_2023_to_Present_2025_enhanced.csv").sample(20000, random_state=42)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Hour'] = df['Date'].dt.hour #type:ignore
    df['Day'] = df['Date'].dt.dayofweek   # 0=Mon → 6=Sun #type:ignore
    df['Month'] = df['Date'].dt.month#type:ignore
    df['IsWeekend'] = df['Day'].apply(lambda x: 1 if x >= 5 else 0)
    return df

df = load_data()


# --------------------------------------
# LOAD BEST KMEANS MODEL FROM MLFLOW
# --------------------------------------
st.sidebar.header("⚙ Model Selection")
st.sidebar.write("Using MLflow saved temporal clustering model.")

model = mlflow.sklearn.load_model("models:/Best_TimeBehavior_Model/latest")#type:ignore

# Scale temporal features for prediction
scaler = StandardScaler()
X = df[['Hour','Day','Month','IsWeekend']]
X_scaled = scaler.fit_transform(X)

# Predict temporal crime clusters
df["TimeCluster"] = model.predict(X_scaled)#type:ignore


# =============================================================
# 📌 INTERPRETATION FUNCTIONS
# =============================================================
def interpret_cluster(hour):
    """Assign human label to peak crime hour."""
    if hour >= 21 or hour <= 2:
        return "Late-Night High-Risk Pattern"
    elif 17 <= hour < 21:
        return "Evening Rush-Hour Pattern"
    elif 8 <= hour < 17:
        return "Daytime Activity Pattern"
    else:
        return "Early Morning Opportunistic Pattern"


def get_season(month):
    if month in [12,1,2]: return "Winter"
    elif month in [3,4,5]: return "Spring"
    elif month in [6,7,8]: return "Summer"
    else: return "Autumn"


df["Season"] = df["Month"].apply(get_season)


# =============================================================
# 📍 HIGH-LEVEL SUMMARY
# =============================================================
st.subheader("📍 Key Temporal Crime Intelligence Summary")

top_hours = df['Hour'].value_counts().nlargest(3)
hour_text = ", ".join([f"{h}:00" for h in top_hours.index])
top_season = df["Season"].value_counts().idxmax()
weekend_size = df[df["IsWeekend"]==1].shape[0]
weekday_size = df[df["IsWeekend"]==0].shape[0]

st.markdown(f"""
- 🔥 **Peak Danger Hours:** `{hour_text}`
- 📅 **Higher Crime Appears On:** `{"Weekends" if weekend_size > weekday_size else "Weekdays"}`
- 🌦 **Season With Most Crimes:** `{top_season}`
""")


# =============================================================
# 📊 TEMPORAL CLUSTER SUMMARY TABLE
# =============================================================
st.subheader("🧠 Time-Based Crime Pattern Clusters (from KMeans)")

day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

cluster_summary = (
    df.groupby("TimeCluster")
    .agg(
        PeakHour=("Hour", lambda x: x.mode()[0]),
        PeakDay=("Day", lambda x: x.mode()[0]),
        PeakMonth=("Month", lambda x: x.mode()[0]),
        TopCrimes=("Primary Type", lambda x: list(x.value_counts().head(3).index))
    )
)

cluster_summary["PeakDay"] = cluster_summary["PeakDay"].map(day_map)
cluster_summary["PeakMonth"] = cluster_summary["PeakMonth"].map(month_map)
cluster_summary["PatternType"] = cluster_summary["PeakHour"].apply(interpret_cluster)

st.dataframe(cluster_summary)


# =============================================================
# 📈 VISUALIZATIONS
# =============================================================

# 🔹 1) Crime by Hour per Cluster
st.subheader("⏰ Crime Volume by Hour")

fig1, ax1 = plt.subplots(figsize=(10,4))
sns.countplot(x=df["Hour"], hue=df["TimeCluster"], palette="tab10", ax=ax1)
st.pyplot(fig1)


# 🔹 2) Weekend vs Weekday Crime
st.subheader("📅 Weekend vs Weekday Crime Pattern")

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.countplot(x=df["IsWeekend"], palette="coolwarm", ax=ax2)
ax2.set_xticklabels(["Weekday", "Weekend"])
st.pyplot(fig2)


# 🔹 3) Monthly Trend
st.subheader("📆 Monthly Crime Trend")

fig3, ax3 = plt.subplots(figsize=(10,4))
sns.countplot(x=df["Month"], hue=df["TimeCluster"], palette="tab10", ax=ax3)
st.pyplot(fig3)


# 🔹 4) Heatmap
st.subheader("🔥 Hourly Heatmap (Crime Intensity Per Cluster)")

heatmap_data = pd.crosstab(df["Hour"], df["TimeCluster"])
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(heatmap_data, cmap="magma", ax=ax4)
st.pyplot(fig4)


# =============================================================
# 🛡 PATROL RECOMMENDATION BRIEFING
# =============================================================
st.subheader("📢 AI-Generated Patrol Scheduling Strategy")

brief = f"""
🕒 **Critical Patrol Time Window:** `{hour_text}`  

👮 Police should increase patrols during:

| Pattern Type | Ideal Patrol Window |
|--------------|---------------------|
| Late-Night Pattern | 11 PM – 3 AM |
| Evening Rush Hour Pattern | 5 PM – 10 PM |
| Daytime Pattern | 9 AM – 5 PM |
| Morning Opportunistic Pattern | 4 AM – 8 AM |

📅 **Weekend Alert:** Weekends require **30–50% more patrol coverage** especially near nightlife and public gathering zones.

🌦 **Season Alert:** `{top_season}` shows peak crime — enable seasonal reinforcement teams.

"""

st.markdown(brief)
