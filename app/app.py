import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="💙 AI Health Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("new_health_anomalies.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['timestamp'].min().date(), df['timestamp'].max().date()]
)
filtered_df = df[(df['timestamp'].dt.date >= date_range[0]) & 
                 (df['timestamp'].dt.date <= date_range[1])]

# -------------------------
# Summary KPIs
# -------------------------
st.title("💙 AI Health Monitoring Dashboard")
st.write("Real-time health anomaly detection with actionable insights.")

total_records = len(filtered_df)
total_anomalies = filtered_df[filtered_df['anomaly']=="Anomaly"].shape[0]
normal_percentage = ((total_records - total_anomalies) / total_records) * 100
latest = filtered_df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", total_records)

with col2:
    st.metric("Anomalies Detected", total_anomalies)

with col3:
    st.metric("Normal (%)", f"{normal_percentage:.1f}%")

with col4:
    status = latest['anomaly']
    st.metric("Latest Status", status, delta_color="inverse" if status=="Anomaly" else "normal")

# -------------------------
# Gauges for Latest Metrics
# -------------------------
st.subheader("🩺 Latest Health Metrics")
g1, g2, g3 = st.columns(3)

with g1:
    fig_hr = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest['heart_rate'],
        delta={'reference': 80},
        title={'text': "Heart Rate (BPM)"},
        gauge={'axis': {'range':[50,150]},
               'bar': {'color': "red" if latest['heart_rate']>100 else "green"}}
    ))
    st.plotly_chart(fig_hr, use_container_width=True)

with g2:
    fig_ox = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['blood_oxygen'],
        title={'text': "Blood Oxygen (%)"},
        gauge={'axis': {'range':[80,100]},
               'bar': {'color': "red" if latest['blood_oxygen']<95 else "green"}}
    ))
    st.plotly_chart(fig_ox, use_container_width=True)

with g3:
    fig_sleep = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['sleep_hours'],
        title={'text': "Sleep Hours"},
        gauge={'axis': {'range':[0,12]},
               'bar': {'color': "orange" if latest['sleep_hours']<6 else "green"}}
    ))
    st.plotly_chart(fig_sleep, use_container_width=True)

# -------------------------
# Trend Charts
# -------------------------
st.subheader("📈 Health Trends Over Time")

fig1 = px.line(
    filtered_df, x='timestamp', y='heart_rate', color='anomaly',
    title="Heart Rate Trend", markers=True
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(
    filtered_df, x='timestamp', y='blood_oxygen', color='anomaly',
    title="Blood Oxygen Trend"
)
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(
    filtered_df, x='timestamp', y='sleep_hours', color='anomaly',
    title="Sleep Hours Trend"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Anomalies Table & Recommendations
# -------------------------
st.subheader("⚠️ Anomalies Detected")
anomaly_df = filtered_df[filtered_df['anomaly']=="Anomaly"]
st.dataframe(anomaly_df)

st.subheader("💡 Recommendations")
for idx, row in anomaly_df.iterrows():
    msg = ""
    if row['heart_rate'] > 100:
        msg += "High heart rate detected. Rest or consult doctor. "
    if row['blood_oxygen'] < 95:
        msg += "Low oxygen level. Breathe deeply or seek help. "
    if row['sleep_hours'] < 6:
        msg += "Insufficient sleep. Prioritize rest. "
    st.warning(f"{row['timestamp']}: {msg}")

# -------------------------
# Download Buttons
# -------------------------
csv = anomaly_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Anomaly Report",
    data=csv,
    file_name="anomaly_report.csv",
    mime="text/csv"
)
