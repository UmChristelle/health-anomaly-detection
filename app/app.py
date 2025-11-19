import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="💙 AI Health Monitoring Dashboard",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS for styling
# -------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    color: #0a3d62;
}
h1 {
    color: #0a3d62;
}
.sidebar .sidebar-content {
    background-color: #dff9fb;
}
.stButton>button {
    background-color: #0a3d62;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.title("💙 AI Health Monitoring Dashboard")
st.markdown("**Real-time health anomaly detection with actionable insights**")

# -------------------------
# Sidebar: Input Options
# -------------------------
st.sidebar.header("Input Options")

input_option = st.sidebar.radio(
    "Choose input method:",
    ("Upload CSV", "Manual Entry")
)

# -------------------------
# Load default data with fallback
# -------------------------
@st.cache_data
def load_default_data():
    # Path relative to app.py
    csv_path = os.path.join(os.path.dirname(__file__), "../new_health_anomalies.csv")
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            st.warning("CSV file is empty. Using demo data.")
    else:
        st.info("CSV file not found. Using demo data.")
    
    # Demo dataset if CSV missing or empty
    data = {
        "timestamp": [datetime.now() - timedelta(hours=i) for i in range(5)],
        "heart_rate": [72, 85, 90, 110, 78],
        "blood_oxygen": [98, 95, 97, 92, 96],
        "sleep_hours": [7, 6, 5, 4, 8],
        "activity_level": ["low", "moderate", "high", "low", "moderate"]
    }
    df = pd.DataFrame(data)
    return df

# -------------------------
# CSV Upload
# -------------------------
if input_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (timestamp, heart_rate, blood_oxygen, sleep_hours, activity_level)",
        type=["csv"]
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
    else:
        st.sidebar.info("Using default/demo data.")
        df = load_default_data()

# -------------------------
# Manual Entry
# -------------------------
if input_option == "Manual Entry":
    st.sidebar.info("Enter your health metrics below:")
    heart_rate = st.sidebar.number_input("Heart Rate (BPM)", min_value=30, max_value=200, value=75)
    blood_oxygen = st.sidebar.number_input("Blood Oxygen (%)", min_value=70, max_value=100, value=98)
    sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    activity_level = st.sidebar.selectbox("Activity Level", ["low", "moderate", "high"])

    df = pd.DataFrame({
        "timestamp": [datetime.now()],
        "heart_rate": [heart_rate],
        "blood_oxygen": [blood_oxygen],
        "sleep_hours": [sleep_hours],
        "activity_level": [activity_level]
    })

# -------------------------
# Ensure timestamp is datetime
# -------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])

# -------------------------
# Preprocessing for anomaly detection
# -------------------------
if 'activity_level' in df.columns:
    activity_dummies = pd.get_dummies(df['activity_level'], prefix='activity_level')
    for col in ['activity_level_low','activity_level_moderate','activity_level_high']:
        if col not in activity_dummies.columns:
            activity_dummies[col] = 0
    df = pd.concat([df, activity_dummies], axis=1)
else:
    df['activity_level_low'] = df['activity_level_moderate'] = df['activity_level_high'] = 0

# -------------------------
# Simple anomaly detection
# -------------------------
def detect_anomaly(row):
    if row['heart_rate'] > 100 or row['blood_oxygen'] < 95 or row['sleep_hours'] < 6:
        return "Anomaly"
    return "Normal"

df['anomaly'] = df.apply(detect_anomaly, axis=1)

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
st.subheader("📊 Summary")
total_records = len(filtered_df)
total_anomalies = filtered_df[filtered_df['anomaly']=="Anomaly"].shape[0]
normal_percentage = ((total_records - total_anomalies) / total_records) * 100
latest = filtered_df.iloc[-1]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Records", total_records, delta_color="off")
kpi2.metric("Anomalies Detected", total_anomalies, delta_color="inverse")
kpi3.metric("Normal (%)", f"{normal_percentage:.1f}%", delta_color="off")
kpi4.metric("Latest Status", latest['anomaly'], delta_color="inverse" if latest['anomaly']=="Anomaly" else "normal")

# -------------------------
# Latest Metrics Gauges
# -------------------------
st.subheader("🩺 Latest Health Metrics")
g1, g2, g3 = st.columns(3)

fig_hr = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=latest['heart_rate'],
    delta={'reference': 80},
    title={'text': "Heart Rate (BPM)"},
    gauge={'axis': {'range':[50,150]},
           'bar': {'color': "red" if latest['heart_rate']>100 else "green"}}
))
st.plotly_chart(fig_hr, use_container_width=True)

fig_ox = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest['blood_oxygen'],
    title={'text': "Blood Oxygen (%)"},
    gauge={'axis': {'range':[80,100]},
           'bar': {'color': "red" if latest['blood_oxygen']<95 else "green"}}
))
st.plotly_chart(fig_ox, use_container_width=True)

fig_sleep = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest['sleep_hours'],
    title={'text': "Sleep Hours"},
    gauge={'axis': {'range':[0,12]},
           'bar': {'color': "orange" if latest['sleep_hours']<6 else "green"}}
))
st.plotly_chart(fig_sleep, use_container_width=True)

# -------------------------
# Health Trends
# -------------------------
st.subheader("📈 Health Trends Over Time")
tab1, tab2, tab3 = st.tabs(["Heart Rate", "Blood Oxygen", "Sleep Hours"])

with tab1:
    fig1 = px.line(filtered_df, x='timestamp', y='heart_rate', color='anomaly', markers=True,
                   title="Heart Rate Trend", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.line(filtered_df, x='timestamp', y='blood_oxygen', color='anomaly',
                   title="Blood Oxygen Trend", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.bar(filtered_df, x='timestamp', y='sleep_hours', color='anomaly',
                  title="Sleep Hours Trend", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Anomalies & Recommendations
# -------------------------
st.subheader("⚠️ Detected Anomalies")
anomaly_df = filtered_df[filtered_df['anomaly']=="Anomaly"]
st.dataframe(anomaly_df.style.highlight_max(axis=0, color='red'))

st.subheader("💡 Recommendations")
for idx, row in anomaly_df.iterrows():
    msg = ""
    if row['heart_rate'] > 100:
        msg += "High heart rate detected. Rest or consult a doctor. "
    if row['blood_oxygen'] < 95:
        msg += "Low oxygen level. Breathe deeply or seek help. "
    if row['sleep_hours'] < 6:
        msg += "Insufficient sleep. Prioritize rest. "
    st.warning(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}: {msg}")

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
