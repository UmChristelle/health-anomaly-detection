import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="AI Health Monitoring Dashboard",
    layout="wide"
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("health_anomalies.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# -------------------------
# Header
# -------------------------
st.title("💙 AI Health Monitoring Dashboard")
st.write("Real-time health anomaly detection using Isolation Forest.")

# -------------------------
# Summary Cards
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Anomalies Detected", df[df['anomaly']=="Anomaly"].shape[0])

with col3:
    normal = df[df['anomaly']=="Normal"].shape[0]
    total = len(df)
    st.metric("Normal (%)", f"{(normal/total)*100:.1f}%")

# -------------------------
# Line Chart: Heart Rate Over Time
# -------------------------
st.subheader("📈 Heart Rate Trend")
fig = px.line(
    df,
    x="timestamp",
    y="heart_rate",
    color="anomaly",
    title="Heart Rate Over Time (Anomalies Highlighted)"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Blood Oxygen Chart
# -------------------------
st.subheader("🩸 Blood Oxygen Levels")
fig2 = px.line(
    df,
    x="timestamp",
    y="blood_oxygen",
    color="anomaly",
    title="Blood Oxygen Over Time"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Sleep Hours Chart
# -------------------------
st.subheader("😴 Sleep Hours")
fig3 = px.bar(
    df,
    x="timestamp",
    y="sleep_hours",
    color="anomaly",
    title="Sleep Duration per Day"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Anomalies Table
# -------------------------
st.subheader("⚠️ Anomaly Records")
anomaly_df = df[df['anomaly']=="Anomaly"]

st.dataframe(anomaly_df)

# -------------------------
# Download Button
# -------------------------
csv = anomaly_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Anomaly Report",
    data=csv,
    file_name="anomaly_report.csv",
    mime="text/csv"
)
