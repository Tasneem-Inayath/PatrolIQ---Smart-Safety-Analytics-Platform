# 🚓 PatrolIQ — Smart Safety Analytics Platform

AI-powered Crime Intelligence System for Urban Safety & Patrol Optimization

---

## 📌 Overview

**PatrolIQ** is an intelligent crime analytics platform built to help law enforcement agencies understand:

✔ **Where** crimes happen
✔ **When** crimes peak
✔ **What types** of crimes occur
✔ **How to deploy patrol units efficiently**

Using **MLflow-tracked models, unsupervised clustering, temporal analysis, dimensionality reduction visualizations, and interactive geospatial maps**, PatrolIQ transforms raw crime records into **actionable insights** for decision-making.

---

## 🎯 Key Goals

* Identify **high-risk crime zones**
* Detect **time-based crime patterns**
* Predict **patrol demand** based on clustering analysis
* Visualize crime severity using **PCA & UMAP embeddings**
* Provide an AI-assisted dashboard for **policing strategy and public safety**

---

## 🧠 Features

| Feature                                        | Description                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| 📍 **Geographic Crime Hotspots**               | Maps using K-Means, DBSCAN & Hierarchical clustering |
| 🚨 **Final DBSCAN Hotspot Model (MLflow)**     | Registered best model predicts top patrol zones      |
| ⏳ **Temporal Crime Pattern Analysis**          | Detect peak crime hours, weekdays, seasons           |
| Ⓜ️ **Dimensionality Reduction (PCA & UMAP)**   | 2D & 3D visual crime pattern visualization           |
| 🖥 **Interactive Streamlit UI**                | Multi-page navigable dashboard                       |
| 🔁 **MLflow Tracking & Model Registry**        | Logged experiments with params, metrics & versions   |
| 📊 **Model Performance & Explainability Page** | Compare trained models and metrics                   |

---

## 📂 Project Structure

```
📁 PatrolIQ
 ┣━━ 🗂 data/
 ┃     ┗━━ Crimes_2023_to_Present_2025_enhanced.csv
 ┣━━ 🗂 mlruns/
 ┣━━ 🗂 pages/
 ┃     ┣━━ Geographic_Hotspots.py
 ┃     ┣━━ Temporal_Analysis.py
 ┃     ┣━━ Dimensionality_Reduction.py
 ┃     ┗━━ Model_Registry_and_Metrics.py
 ┣━━ Home.py
 ┣━━ requirements.txt
 ┣━━ README.md
```

---

## 🚀 Technology Stack

| Category                 | Tools                               |
| ------------------------ | ----------------------------------- |
| Programming              | Python                              |
| Web Framework            | Streamlit                           |
| ML Tracking              | MLflow                              |
| Clustering Algorithms    | KMeans, DBSCAN, Agglomerative       |
| Dimensionality Reduction | PCA, UMAP                           |
| Visualization            | Folium, seaborn, matplotlib, plotly |
| Deployment               | Streamlit Cloud, GitHub             |

---

## 🧪 Machine Learning Models Logged to MLflow

| Model Name           | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `KMeans_Model_Geo`   | Clusters crime spatially                 |
| `DBSCAN_Model`       | Detects true hotspot density             |
| `Hierarchical_Model` | Multiscale crime zone relationships      |
| `Temporal_KMeans`    | Classifies daily/seasonal crime behavior |
| `PCA_Model`          | Feature reduction for visualization      |
| `UMAP_Model`         | High-resolution crime pattern mapping    |

---

## 🔥 Insights Generated

✔ Top 5 high-risk patrol zones using density-based clustering
✔ Peak crime timing (midnight spike, weekend surge, seasonal variation)
✔ Crime type behavior cluster (nightlife-related thefts vs daytime burglary)
✔ Interactive PCA and UMAP embedders revealing crime pattern structure

---
##Link for Review:
[Link](https://patroliq---smart-safety-analytics-platform-hfr7rzquo98x5pzdg2s.streamlit.app/)
---
## 📌 Deployment Guide

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run MLflow (optional)

```
mlflow ui
```

### 3️⃣ Launch App

```
streamlit run Home.py
```

---

## 🧿 Future Enhancements

* 🔍 Real-time streaming crime prediction
* 🤖 LSTM-based temporal forecasting
* 🗺️ Route optimization for patrol vehicles
* 📱 Police mobile app integration

---

## ❤️ Acknowledgment

Built with **passion, patience, and data science curiosity** 💛
This project is dedicated to helping improve **public safety and intelligent policing.**

---

## 🧕 Author — Tasneem Inayath

If you like the project, ⭐ **star the repository** and share feedback!
