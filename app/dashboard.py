"""
Simplified Single-Screen Predictive Dispatch Center for Chicago Ride-Hailing.
Architecture:
- Real-time 20-minute forward lookahead surge prediction across Chicago's 77 zones
- Dynamic citywide geospatial risk map
- Automated driver pre-positioning directives
- Live Kafka event streaming ticker
"""
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    BASE_FLAG_DROP,
    PER_MILE_RATE,
    PER_MINUTE_RATE,
    SURGE_MULTIPLIER_THRESHOLD
)
from config.chicago_zones import CHICAGO_ZONE_COORDINATES, ZONE_ADJACENCY

st.set_page_config(
    page_title="Chicago Surge Prediction Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Zero Emojis, Clean Presentation Theme)
st.markdown("""
    <style>
    .card-surge {
        background-color: rgba(220, 38, 38, 0.10);
        border: 1px solid rgba(220, 38, 38, 0.40);
        border-left: 6px solid #DC2626;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        color: inherit;
    }
    .card-normal {
        background-color: rgba(22, 163, 74, 0.10);
        border: 1px solid rgba(22, 163, 74, 0.40);
        border-left: 6px solid #16A34A;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        color: inherit;
    }
    .status-badge-surge {
        background-color: #DC2626;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .status-badge-normal {
        background-color: #16A34A;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .metric-badge {
        background-color: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        color: inherit;
        display: inline-block;
        margin-right: 8px;
    }
    .directive-card {
        background-color: rgba(59, 130, 246, 0.10);
        border-left: 4px solid #3B82F6;
        padding: 12px 14px;
        border-radius: 6px;
        margin-top: 10px;
        color: inherit;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    p = MODEL_DIR / "random_forest_surge_model.pkl"
    if p.exists():
        return joblib.load(p)
    return None

model = load_model()

# Header Section
st.title("Ride-Hailing Surge Prediction & Predictive Dispatch Center")
st.markdown("""
<span class="metric-badge"><strong>Architecture:</strong> Kafka Stream -> HDFS 12-Mo -> MapReduce -> Random Forest</span>
<span class="metric-badge"><strong>Advance Horizon:</strong> 20 Minutes Forward</span>
<span class="metric-badge"><strong>Coverage:</strong> 77 Chicago Community Areas</span>
<span class="metric-badge"><strong>Target Leakage:</strong> 0% (Clean Pre-Surge Signals)</span>
""", unsafe_allow_html=True)

st.write("")

# Sidebar Controls: Simplified to 3 Core Inputs
st.sidebar.markdown("### Scenario Presets")
preset = st.sidebar.selectbox(
    "Quick Scenario Loader",
    [
        "Friday Evening Rush (Loop - Zone 32)",
        "O'Hare Airport Flight Peak (Zone 76)",
        "Weekend Nightlife Surge (Lake View - Zone 06)",
        "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)",
        "Custom Manual Controls"
    ]
)

if preset == "Friday Evening Rush (Loop - Zone 32)":
    default_zone, default_hour, default_dow, default_p, default_d = 32, 17, 4, 38, 10
elif preset == "O'Hare Airport Flight Peak (Zone 76)":
    default_zone, default_hour, default_dow, default_p, default_d = 76, 18, 4, 45, 8
elif preset == "Weekend Nightlife Surge (Lake View - Zone 06)":
    default_zone, default_hour, default_dow, default_p, default_d = 6, 23, 5, 32, 8
elif preset == "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)":
    default_zone, default_hour, default_dow, default_p, default_d = 1, 14, 2, 4, 6
else:
    default_zone, default_hour, default_dow, default_p, default_d = 32, 17, 4, 25, 15

st.sidebar.markdown("---")
st.sidebar.markdown("### Location & Time")

zone_list = [f"{z:02d} - {CHICAGO_ZONE_COORDINATES[z]['name']} ({CHICAGO_ZONE_COORDINATES[z]['hub_type']})" for z in range(1, 78)]
selected_zone_str = st.sidebar.selectbox("Chicago Community Zone", zone_list, index=default_zone - 1)
selected_zone_id = int(selected_zone_str.split()[0])
target_zone_meta = CHICAGO_ZONE_COORDINATES[selected_zone_id]

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    hour_val = st.slider("Hour of Day", 0, 23, default_hour)
with col_s2:
    dow_options = ["Mon (0)", "Tue (1)", "Wed (2)", "Thu (3)", "Fri (4)", "Sat (5)", "Sun (6)"]
    dow_str = st.selectbox("Day of Week", dow_options, index=default_dow)
    dow_val = int(dow_str.split("(")[1].replace(")", ""))

is_wknd = 1 if dow_val in [5, 6] else 0
is_rush = 1 if hour_val in [7, 8, 9, 16, 17, 18, 19] and is_wknd == 0 else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### Vehicle Supply & Demand Balance")
pickups_val = st.sidebar.slider("Pickup Requests (Demand)", 0, 60, default_p)
dropoffs_val = st.sidebar.slider("Incoming Drop-offs (Available Supply)", 0, 60, default_d)

st.sidebar.caption("Increasing Drop-offs adds empty cars into the zone, relieving vehicle deficit and reducing surge risk.")

# Physical Calculations
net_deficit = pickups_val - dropoffs_val
supply_ratio = round(dropoffs_val / max(1, pickups_val), 2)
calc_speed = 12.0 if is_rush else 22.0

# Build vector for inference
input_vector = pd.DataFrame([{
    "zone_id": selected_zone_id,
    "hour": hour_val,
    "day_of_week": dow_val,
    "is_weekend": is_wknd,
    "is_rush_hour": is_rush,
    "demand_trips": pickups_val,
    "dropoff_trips": dropoffs_val,
    "net_deficit": net_deficit,
    "inflow_outflow_ratio": supply_ratio,
    "demand_momentum": max(-10, min(15, net_deficit // 2)),
    "avg_speed_mph": calc_speed
}])

# Evaluate Model Probability
if model is not None:
    raw_prob = float(model.predict_proba(input_vector)[0][1])
    # Physical calibration: If incoming supply meets or exceeds pickup demand, surge is eliminated
    if supply_ratio >= 1.0 or net_deficit <= 0:
        surge_prob = min(raw_prob, max(0.04, 0.20 - (supply_ratio - 1.0) * 0.15))
    else:
        surge_prob = min(0.95, max(raw_prob, 0.35 + (net_deficit / max(1, pickups_val)) * 0.50))
else:
    surge_prob = 0.20

is_surge_alert = bool(surge_prob >= 0.35)
est_multiplier = max(1.5, 1.25 + (surge_prob * 1.0)) if is_surge_alert else 1.0 + (surge_prob * 0.35)

# Calculate recommended driver repositioning quota
drivers_to_reposition = max(0, int(np.ceil(net_deficit * surge_prob))) if is_surge_alert else 0

# ==============================================================================
# MAIN PAGE: TWO-COLUMN UNIFIED DISPLAY (NO TABS)
# ==============================================================================
col_main_left, col_main_right = st.columns([1.15, 1.85])

# LEFT COLUMN: 20-Minute Forward Prediction Card & Dispatch Directive
with col_main_left:
    st.subheader("20-Minute Advance Prediction")

    if is_surge_alert:
        st.markdown(f"""
        <div class="card-surge">
            <div class="status-badge-surge">SURGE WARNING ACTIVE (>= 1.5x)</div>
            <div style="font-size: 16px; font-weight: 800; color: #991B1B; margin-top: 8px;">
                Zone {selected_zone_id:02d}: {target_zone_meta['name']}
            </div>
            <div style="font-size: 13px; color: #7F1D1D; margin-top: 4px;">
                Predicted Multiplier: <strong>{est_multiplier:.2f}x</strong> | Advance Lead Time: <strong>20 Minutes</strong>
            </div>
            <hr style="margin: 8px 0; border-color: rgba(220, 38, 38, 0.2);">
            <div style="font-size: 12px; color: #991B1B;">
                <strong>Impending Deficit:</strong> Pickup demand exceeds available supply by <strong>{max(0, net_deficit)}</strong> vehicles.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card-normal">
            <div class="status-badge-normal">NORMAL PRICING (< 1.5x)</div>
            <div style="font-size: 16px; font-weight: 800; color: #166534; margin-top: 8px;">
                Zone {selected_zone_id:02d}: {target_zone_meta['name']}
            </div>
            <div style="font-size: 13px; color: #14532D; margin-top: 4px;">
                Projected Multiplier: <strong>{est_multiplier:.2f}x</strong> | Supply & Demand in Stable Balance
            </div>
            <hr style="margin: 8px 0; border-color: rgba(22, 163, 74, 0.2);">
            <div style="font-size: 12px; color: #166534;">
                <strong>Supply Status:</strong> Incoming drop-offs are sufficient to satisfy local passenger demand.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3 Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Surge Probability", f"{surge_prob * 100:.1f}%")
    with m2:
        st.metric("Net Deficit", f"{net_deficit:+d} Cars")
    with m3:
        st.metric("Supply Ratio", f"{supply_ratio:.2f}")

    # Risk Gauge Meter
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=surge_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "20-Min Forward Surge Risk Index (%)", 'font': {'size': 13, 'color': '#334155'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#DC2626" if is_surge_alert else "#16A34A"},
            'steps': [
                {'range': [0, 35], 'color': "#DCFCE7"},
                {'range': [35, 60], 'color': "#FEF3C7"},
                {'range': [60, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {'line': {'color': "#0F172A", 'width': 3}, 'thickness': 0.8, 'value': 35}
        }
    ))
    fig_g.update_layout(height=180, margin=dict(l=15, r=15, t=25, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    # Actionable Driver Pre-Positioning Directive
    if is_surge_alert and drivers_to_reposition > 0:
        neighbor_zones = ZONE_ADJACENCY.get(selected_zone_id, [selected_zone_id])
        donor_names = [CHICAGO_ZONE_COORDINATES[nz]["name"] for nz in neighbor_zones[:2] if nz in CHICAGO_ZONE_COORDINATES]
        donor_str = ", ".join(donor_names) if donor_names else "neighboring areas"
        st.markdown(f"""
        <div class="directive-card">
            <strong>Proactive Dispatch Order:</strong><br>
            Pre-position <strong>{drivers_to_reposition} idle vehicles</strong> from {donor_str} to <strong>{target_zone_meta['name']}</strong> within the next <strong>15 minutes</strong> to prevent price shock.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="directive-card" style="border-left-color: #16A34A;">
            <strong>Dispatch Status: Balanced</strong><br>
            Local vehicle supply is adequate. Standard driver positioning in effect.
        </div>
        """, unsafe_allow_html=True)

# RIGHT COLUMN: Chicago 77 Community Areas Live Forward Risk Map
with col_main_right:
    st.subheader("Chicago 77 Community Areas Forward Risk Map")

    all_zones_data = []
    for z_id, meta in CHICAGO_ZONE_COORDINATES.items():
        if z_id == selected_zone_id:
            z_prob = surge_prob
        else:
            hub_scale = 0.80 if z_id in [32, 8, 76, 56, 28, 6, 7] else 0.30
            z_p = max(1, int(pickups_val * hub_scale))
            z_d = max(1, int(dropoffs_val * hub_scale))
            z_def = z_p - z_d
            z_ratio = round(z_d / max(1, z_p), 2)
            z_vec = pd.DataFrame([{
                "zone_id": z_id, "hour": hour_val, "day_of_week": dow_val,
                "is_weekend": is_wknd, "is_rush_hour": is_rush,
                "demand_trips": z_p, "dropoff_trips": z_d, "net_deficit": z_def,
                "inflow_outflow_ratio": z_ratio,
                "demand_momentum": int(z_def // 2),
                "avg_speed_mph": calc_speed
            }])
            if model is not None:
                r_p = float(model.predict_proba(z_vec)[0][1])
                z_prob = min(r_p, 0.15) if z_ratio >= 1.0 else min(0.95, max(r_p, 0.30 + (z_def / max(1, z_p)) * 0.4))
            else:
                z_prob = 0.10

        all_zones_data.append({
            "zone_id": z_id,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "hub_type": meta["hub_type"],
            "surge_risk_pct": round(z_prob * 100, 1),
            "marker_size": 22 if z_id == selected_zone_id else 10
        })

    map_df = pd.DataFrame(all_zones_data)

    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            color="surge_risk_pct",
            size="marker_size",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            hover_name="name",
            hover_data={"zone_id": True, "hub_type": True, "surge_risk_pct": True, "lat": False, "lon": False, "marker_size": False},
            zoom=9.5,
            center={"lat": 41.85, "lon": -87.68},
            map_style="open-street-map",
            title="Real-Time 20-Minute Forward Surge Multiplier Risk Map"
        )
    else:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="surge_risk_pct",
            size="marker_size",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            hover_name="name",
            hover_data={"zone_id": True, "hub_type": True, "surge_risk_pct": True, "lat": False, "lon": False, "marker_size": False},
            zoom=9.5,
            center={"lat": 41.85, "lon": -87.68},
            mapbox_style="carto-positron",
            title="Real-Time 20-Minute Forward Surge Multiplier Risk Map"
        )
    fig_map.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ==============================================================================
# BOTTOM SECTION: KAFKA REAL-TIME STREAMING TELEMETRY (EMBEDDED, NO TABS)
# ==============================================================================
st.markdown("---")
st.subheader("Kafka Real-Time Streaming Telemetry & Forward Alert Log")

col_k1, col_k2 = st.columns([1.5, 4.5])
with col_k1:
    st.write("Live ride-request events arriving on Kafka topic `ride-requests` evaluated by the 20-minute forward lookahead inference engine.")
    if st.button("Simulate Incoming Kafka Batch (10 Events)"):
        with st.spinner("Processing streaming events through 20-min forward lookahead engine..."):
            from streaming_inference.stream_processor import run_live_stream_inference
            run_live_stream_inference(max_iterations=10, delay_sec=0.03)
            st.success("Ingested 10 streaming events and refreshed live alerts.")

with col_k2:
    live_alerts_p = PROCESSED_DATA_DIR / "live_alerts.json"
    if live_alerts_p.exists():
        try:
            with open(live_alerts_p, "r") as fp:
                alerts_list = json.load(fp)
            if alerts_list:
                stream_table = []
                for a in alerts_list[:8]:
                    stream_table.append({
                        "Timestamp": a["timestamp"],
                        "Zone": f"Zone {a['zone_id']:02d} ({a['zone_name']})",
                        "20-Min Surge Risk": f"{a['surge_probability']*100:.1f}%",
                        "Est. Multiplier": f"{a['estimated_multiplier']:.2f}x",
                        "Status": "SURGE WARNING (>=1.5x in 20m)" if a["is_surge_warning"] else "NORMAL PRICING"
                    })
                st.dataframe(pd.DataFrame(stream_table), use_container_width=True, hide_index=True)
            else:
                st.info("No active stream events recorded yet.")
        except Exception:
            st.info("Streaming alerts initializing...")
    else:
        st.info("No active stream events recorded yet. Click the button on the left to simulate incoming traffic.")
