import io 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Data Analysis & Feature Engineering", layout="wide")
st.title("📊 Data Analysis & Feature Engineering")


# --- Load Uncleaned Dataset ---
@st.cache_data
def load_uncleaned(file_path):
    return pd.read_csv(file_path)

uncleaned_data = load_uncleaned("data/Crimes_2023_to_Present_2025.csv")
st.subheader("Original (Uncleaned) Dataset")
st.write("**Shape:**", uncleaned_data.shape)
st.dataframe(uncleaned_data.head(20))

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data("data/Crimes_2023_to_2025_CLEANED.csv")
st.subheader("✅Cleaned Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(10))

# --------------------------
# EDA Section
# --------------------------
st.subheader("🔍 Exploratory Data Analysis")

# Scatter plot of locations
st.markdown("**Crime Locations Scatter Plot**")
fig, ax = plt.subplots()
ax.scatter(df['Longitude'], df['Latitude'], s=1, alpha=0.4, color='purple')
ax.set_title("Crime Locations in Chicago")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# Crime type distribution bar chart
st.markdown("**Crime Distribution Across Types**")
crime_count = df['Primary Type'].value_counts()
fig, ax = plt.subplots(figsize=(12,8))
crime_count.sort_values(ascending=True).plot(kind='barh', color='skyblue', ax=ax)
ax.set_title('Crime Distribution Across 33 Crime Types', fontsize=14)
ax.set_xlabel('Number of Reported Crimes')
ax.set_ylabel('Crime Type')
st.pyplot(fig)

# Top 10 Crime types pie chart
st.markdown("**Top 10 Crime Types by Percentage**")
top_10 = (crime_count / crime_count.sum()).head(10)
fig, ax = plt.subplots(figsize=(8,8))
ax.pie(top_10, labels=top_10.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Top 10 Crime Types by Percentage')
st.pyplot(fig)

# Crimes by Hour
st.markdown("**Crimes by Hour of Day**")
fig, ax = plt.subplots(figsize=(10,5))
df['Hour'].value_counts().sort_index().plot(kind='bar', color='teal', ax=ax)
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Crimes")
ax.set_title("Crimes by Hour of Day")
st.pyplot(fig)

# Crimes by Day of Week
st.markdown("**Crimes by Day of Week**")
fig, ax = plt.subplots(figsize=(8,5))
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['Day_of_Week'].value_counts().reindex(order).plot(kind='bar', color='orange', ax=ax)
ax.set_ylabel("Number of Crimes")
ax.set_title("Crimes by Day of Week")
st.pyplot(fig)

# Crimes by Month
st.markdown("**Crimes by Month**")
fig, ax = plt.subplots(figsize=(10,5))
month_order = list(range(1,13))
df['Month'].value_counts().reindex(month_order).plot(kind='bar', color='green', ax=ax)
ax.set_ylabel("Number of Crimes")
ax.set_title("Crimes by Month")
st.pyplot(fig)

# Arrest rate by Domestic
st.markdown("**Arrest Rate: Domestic vs Non-Domestic Crimes**")
fig, ax = plt.subplots(figsize=(5,4))
df.groupby('Domestic')['Arrest'].mean().plot(kind='bar', color=['pink','violet'], ax=ax)
ax.set_xticklabels(['Non-Domestic','Domestic'], rotation=0)
ax.set_ylabel("Arrest Rate (%)")
ax.set_title("Arrest Rate: Domestic vs Non-Domestic")
st.pyplot(fig)

# --------------------------
# Feature Engineering Section
# --------------------------
st.subheader("🛠 Feature Engineering")

# Season mapping
st.markdown("**Season Column Example**")
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12,1,2] else 
                                 'Spring' if x in [3,4,5] else 
                                 'Summer' if x in [6,7,8] else 'Fall')
st.dataframe(df[['Month','Season']].head(10))

# Geo Clustering
st.markdown("**Geo Clustering Example**")
from sklearn.cluster import KMeans
coords = df[['Latitude','Longitude']]
kmeans = KMeans(n_clusters=10, random_state=42)
df['Geo_Cluster'] = kmeans.fit_predict(coords)
fig, ax = plt.subplots(figsize=(10,8))
scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['Geo_Cluster'], cmap='tab10', s=10, alpha=0.6)
fig.colorbar(scatter, ax=ax, label="Geo Cluster")
ax.set_title("Crime Locations by Geo Cluster")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# Coordinate binning
df['Lat_Bin'] = pd.cut(df['Latitude'], bins=10, labels=False)
df['Lon_Bin'] = pd.cut(df['Longitude'], bins=10, labels=False)
st.dataframe(df[['Latitude','Longitude','Lat_Bin','Lon_Bin']].head(10))

# Crime Severity
st.markdown("**Crime Severity Feature**")
severity_mapping = {
    'HOMICIDE': 5, 'KIDNAPPING': 5, 'CRIMINAL SEXUAL ASSAULT': 5,
    'SEX OFFENSE': 4, 'ARSON': 4, 'STALKING': 4, 'OFFENSE INVOLVING CHILDREN': 5,
    'PROSTITUTION': 3, 'CONCEALED CARRY LICENSE VIOLATION': 3,
    'LIQUOR LAW VIOLATION': 2, 'INTIMIDATION': 4, 'ROBBERY': 4,
    'BURGLARY': 4, 'WEAPONS VIOLATION': 4, 'NARCOTICS': 3,
    'CRIMINAL TRESPASS': 3, 'BATTERY': 2, 'ASSAULT': 3, 'CRIMINAL DAMAGE': 2,
    'DECEPTIVE PRACTICE': 2, 'OTHER OFFENSE': 2, 'MOTOR VEHICLE THEFT': 3,
    'THEFT': 1, 'PUBLIC PEACE VIOLATION': 2, 'INTERFERENCE WITH PUBLIC OFFICER': 3,
    'OBSCENITY': 1, 'GAMBLING': 1, 'HUMAN TRAFFICKING': 5, 'PUBLIC INDECENCY': 1,
    'OTHER NARCOTIC VIOLATION': 2, 'NON-CRIMINAL': 1
}
df['Crime_Severity'] = df['Primary Type'].map(severity_mapping)
st.dataframe(df[['Primary Type','Crime_Severity']].head(10))

# Label Encoding example
st.markdown("**Label Encoding Examples**")
le = LabelEncoder()
df['Primary_Type_Code'] = le.fit_transform(df['Primary Type'])
df['Location_Code'] = le.fit_transform(df['Location Description'])
st.dataframe(df[['Primary Type','Primary_Type_Code','Location Description','Location_Code']].head(10))

# Frequency Encoding
st.markdown("**Location Frequency Encoding Example**")
location_freq = df['Location Description'].value_counts().to_dict()
df['Location_Freq'] = df['Location Description'].map(location_freq)
st.dataframe(df[['Location Description','Location_Freq']].head(10))

# StandardScaler example
st.markdown("**Scaled Coordinates Example**")
scaler = StandardScaler()
df[['Latitude_scaled','Longitude_scaled']] = scaler.fit_transform(df[['Latitude','Longitude']])
fig, ax = plt.subplots(figsize=(12,5))
ax.scatter(df['Longitude'], df['Latitude'], alpha=0.6, c='skyblue', edgecolor='k', label='Original')
ax.scatter(df['Longitude_scaled'], df['Latitude_scaled'], alpha=0.6, c='orange', edgecolor='k', label='Scaled')
ax.set_title("Original vs Scaled Geographic Coordinates")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)

st.success("✅ EDA and Feature Engineering Displayed Successfully!")

# --- Optional: Show summary info ---
st.subheader("Cleaned Dataset Info")
info_table = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": df.notnull().sum().values,
    "Dtype": df.dtypes.values
})

st.dataframe(info_table)


st.subheader("Summary Statistics of Cleaned Data")
st.write(df.describe())
