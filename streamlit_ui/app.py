import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sqlalchemy import create_engine
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Machinery Utilization Dashboard", layout="wide")
st.title("Machinery Utilization Dashboard")

def get_engine():
    db_host = os.getenv("DB_HOST", "localhost")
    db_user = os.getenv("DB_USER", "eagle")
    db_pass = os.getenv("DB_PASS", "eaglepass")
    db_name = os.getenv("DB_NAME", "equipmentdb")
    return create_engine(
        f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
    )

def load_latest():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT DISTINCT ON (equipment_id)
                equipment_id, equipment_class,
                current_state, current_activity, motion_source,
                total_tracked_seconds, total_active_seconds,
                total_idle_seconds, utilization_percent, time
            FROM equipment_events
            ORDER BY equipment_id, time DESC
        """, conn)
    return df

# Auto refresh every 2000ms (2 seconds)
# This is much gentler than st.rerun() in a tight loop
# The page refreshes cleanly without freezing the browser
st_autorefresh(interval=7000, key="dashboard_refresh")

# ── VIDEO FEED ────────────────────────────────────────────────────
st.subheader("📹 Live Video Feed")
st.components.v1.iframe(
    src="http://localhost:5000/video",
    height=500,
    scrolling=False
)

st.divider()

# ── ANALYTICS ─────────────────────────────────────────────────────
st.subheader("📊 Equipment Analytics")

try:
    df = load_latest()

    if df.empty:
        st.warning("No data yet. Waiting for CV service...")
    else:
        for _, row in df.iterrows():
            st.subheader(f"🚜 {row['equipment_id']} ({row['equipment_class']})")

            col1, col2, col3, col4 = st.columns(4)
            state_color = "🟢" if row["current_state"] == "ACTIVE" else "🔴"
            col1.metric("Status",        f"{state_color} {row['current_state']}")
            col2.metric("Activity",      row["current_activity"])
            col3.metric("Utilization",   f"{row['utilization_percent']}%")
            col4.metric("Motion Source", row["motion_source"])

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("⏱ Total Tracked", f"{row['total_tracked_seconds']:.1f}s")
            col_b.metric("✅ Active Time",   f"{row['total_active_seconds']:.1f}s")
            col_c.metric("💤 Idle Time",     f"{row['total_idle_seconds']:.1f}s")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_pie = px.pie(
                    values=[row["total_active_seconds"], row["total_idle_seconds"]],
                    names=["Active", "Idle"],
                    color_discrete_sequence=["#00cc96", "#ef553b"],
                    title=f"{row['equipment_id']} — Time Breakdown")
                
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                fig_bar = px.bar(
                    x=["Utilization"],
                    y=[row["utilization_percent"]],
                    color=["Utilization"],
                    color_discrete_sequence=["#636efa"],
                    title=f"{row['equipment_id']} — Utilization %",
                    range_y=[0, 100]
                )
                fig_bar.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Target 70%"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

except Exception as e:
    st.error(f"Database error: {e}")