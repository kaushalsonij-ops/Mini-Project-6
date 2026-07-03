import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Title
st.set_page_config(page_title="Customer Segmentation", layout="centered")
# Page styling
st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.write("Hello I am Kaushal")

# Data Load
df = pd.read_csv("Customerdata.csv")
# Data preview
with st.expander("📂 Data Preview"):
    st.dataframe(df)

# Changed: Age, Income instead of Study_Hours, Attendance
x = df[["Age","Income"]]

k = st.slider("Enter no. of clusters", min_value=2, max_value=40, value=3, step=1)

model = KMeans(n_clusters=k, random_state=42, n_init=40)

df["Cluster"] = model.fit_predict(x)

st.subheader("✅ Clustered Data")
st.dataframe(df)

# Changed: Age, Income columns
centers = pd.DataFrame(model.cluster_centers_, columns=["Age","Income"])

st.subheader("📍 Cluster Centers")
st.dataframe(centers.round(2))

st.subheader("📊 Customer Clusters Graph")
fig, ax = plt.subplots(figsize=(6, 4))

# Changed: Age vs Income plot
scatter = ax.scatter(df["Age"], df["Income"], c=df["Cluster"], cmap="viridis", s=100, alpha=0.8, edgecolors='black')

# Cluster Centers
ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c="green", marker="*", s=100, label="Cluster Centers", edgecolors='black')

# Graph Requirements
ax.set_title("Age vs Income - Customer Segments", fontsize=15, fontweight='bold')
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income")
ax.grid(True)
ax.legend()


st.pyplot(fig)