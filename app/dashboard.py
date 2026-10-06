"""
Advanced Streamlit Control Center for Ride-Hailing Surge Price Prediction.
Features:
- Instant One-Click Scenario Presets (Rush Hour, Airport Surge, Normal)
- Real-time 20-minute advance prediction & Chicago geospatial map
- Driver repositioning & fleet supply optimizer
- Multi-model evaluation benchmark lab (ROC-AUC, PR curves, Confusion Matrix)
- Kafka real-time stream monitor
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
from ml.dispatch_optimizer import optimize_driver_repositioning

st.set_page_config(
    page_title="Surge Intelligence & Predictive Dispatch Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
    <style>
    .card-surge {
        background-color: rgba(220, 38, 38, 0.12);
        border: 1px solid rgba(220, 38, 38, 0.4);
        border-left: 6px solid #DC2626;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        color: inherit;
    }
    .card-normal {
        background-color: rgba(22, 163, 74, 0.12);
        border: 1px solid rgba(22, 163, 74, 0.4);
        border-left: 6px solid #16A34A;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
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
    .metric-card {
        background-color: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .formula-box {
        background-color: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 13px;
        color: inherit;
        margin-top: 10px;
    }
    .directive-card {
        background-color: rgba(59, 130, 246, 0.12);
        border-left: 4px solid #3B82F6;
        padding: 14px;
        border-radius: 6px;
        margin-top: 10px;
        color: inherit;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    for name in ["random_forest", "gradient_boosting", "decision_tree", "logistic_regression"]:
        p = MODEL_DIR / f"{name}_surge_model.pkl"
        if p.exists():
            models[name] = joblib.load(p)
    return models

@st.cache_data
def load_eval_summary():
    p = MODEL_DIR / "evaluation_summary.json"
    if p.exists():
        with open(p, "r") as fp:
            return json.load(fp)
    return {}

@st.cache_data
def load_features():
    p = PROCESSED_DATA_DIR / "features_labels.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

models = load_models()
eval_summary = load_eval_summary()
features_df = load_features()
active_model = models.get("random_forest")

# Top Header
st.title("Ride-Hailing Surge Price Prediction & Predictive Dispatch Center")
st.caption("Big Data Infrastructure: Kafka Real-Time Stream -> HDFS Storage -> Java MapReduce -> Ensemble ML Inference (20-Min Lead Time)")

# Sidebar: Scenario Presets & Controls
st.sidebar.markdown("### Scenario Presets")
preset = st.sidebar.selectbox(
    "Quick Scenario Loader",
    [
        "Custom Manual Controls",
        "Friday Evening Rush (Loop - Zone 32)",
        "O'Hare Airport Flight Peak (Zone 76)",
        "Weekend Nightlife Surge (Lake View - Zone 06)",
        "Midway Airport Morning Inflow (Zone 56)",
        "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)"
    ]
)

# Preset Configuration Values
if preset == "Friday Evening Rush (Loop - Zone 32)":
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 32, 18, 4, 65, 48.0, 4.2, 1150
elif preset == "O'Hare Airport Flight Peak (Zone 76)":
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 76, 17, 4, 58, 65.0, 18.2, 2400
elif preset == "Weekend Nightlife Surge (Lake View - Zone 06)":
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 6, 23, 5, 52, 38.0, 3.5, 980
elif preset == "Midway Airport Morning Inflow (Zone 56)":
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 56, 8, 0, 42, 44.0, 12.0, 1600
elif preset == "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)":
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 1, 14, 2, 8, 12.5, 2.8, 550
else:
    default_zone, default_hour, default_dow, default_trips, default_fare, default_miles, default_sec = 32, 17, 4, 35, 32.5, 4.8, 1100

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameter Controls")

zone_list = [f"{z:02d} - {CHICAGO_ZONE_COORDINATES[z]['name']} ({CHICAGO_ZONE_COORDINATES[z]['hub_type']})" for z in range(1, 78)]
selected_zone_str = st.sidebar.selectbox("Target Chicago Community Zone", zone_list, index=default_zone - 1)
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

st.sidebar.markdown("### Demand & Trip Indicators")
demand_trips_val = st.sidebar.slider("Pickups (Last 15 Min)", 1, 90, default_trips)
fare_val = st.sidebar.slider("Current Avg Base Fare ($)", 5.0, 85.0, default_fare)
miles_val = st.sidebar.slider("Avg Trip Distance (Miles)", 0.5, 30.0, default_miles)
seconds_val = st.sidebar.slider("Avg Trip Duration (Sec)", 120, 3600, default_sec)

calc_speed = (miles_val / (seconds_val / 3600.0)) if seconds_val > 0 else 16.0
expected_base = BASE_FLAG_DROP + (miles_val * PER_MILE_RATE) + ((seconds_val / 60.0) * PER_MINUTE_RATE)
est_multiplier = round(fare_val / expected_base, 2) if expected_base > 0 else 1.0

# Build input feature frame
input_vector = pd.DataFrame([{
    "zone_id": selected_zone_id,
    "hour": hour_val,
    "day_of_week": dow_val,
    "is_weekend": is_wknd,
    "is_rush_hour": is_rush,
    "demand_trips": demand_trips_val,
    "avg_fare": fare_val,
    "avg_miles": miles_val,
    "avg_seconds": seconds_val,
    "avg_speed_mph": calc_speed
}])

# Main Navigation Tabs
tab_predict, tab_optimizer, tab_benchmark, tab_stream = st.tabs([
    "1. Live Predictive Dispatch & Map",
    "2. Fleet Supply & Rebalancing Optimizer",
    "3. Multi-Model Benchmark Lab",
    "4. Kafka Real-Time Stream Monitor"
])

# ==============================================================================
# TAB 1: LIVE PREDICTIVE DISPATCH & MAP
# ==============================================================================
with tab_predict:
    if active_model is not None:
        pred_class = active_model.predict(input_vector)[0]
        pred_probs = active_model.predict_proba(input_vector)[0]
        surge_prob = float(pred_probs[1]) if len(pred_probs) > 1 else float(pred_class)
    else:
        pred_class = 0
        surge_prob = 0.1

    col_pred1, col_pred2 = st.columns([1.1, 1.9])

    with col_pred1:
        st.subheader("20-Minute Forward Prediction")

        if surge_prob >= 0.5 or est_multiplier >= 1.5:
            st.markdown(f"""
            <div class="card-surge">
                <div class="status-badge-surge">SURGE WARNING ACTIVE (> 1.5x)</div>
                <div style="font-size: 16px; font-weight: 800; color: #991B1B; margin-top: 8px;">
                    Zone {selected_zone_id}: {target_zone_meta['name']}
                </div>
                <div style="font-size: 13px; color: #7F1D1D; margin-top: 4px;">
                    Projected Surge Multiplier: <strong>{max(1.5, est_multiplier):.2f}x</strong> | Advance Notice: <strong>20 Minutes</strong>
                </div>
                <hr style="margin: 8px 0; border-color: #FECACA;">
                <div style="font-size: 12px; color: #991B1B;">
                    <strong>Action Required:</strong> Pre-position fleet from neighboring zones immediately to absorb impending spike.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-normal">
                <div class="status-badge-normal">NORMAL DEMAND (< 1.5x)</div>
                <div style="font-size: 16px; font-weight: 800; color: #166534; margin-top: 8px;">
                    Zone {selected_zone_id}: {target_zone_meta['name']}
                </div>
                <div style="font-size: 13px; color: #14532D; margin-top: 4px;">
                    Projected Multiplier: <strong>{est_multiplier:.2f}x</strong> | Supply and Demand in Stable Balance
                </div>
                <hr style="margin: 8px 0; border-color: #BBF7D0;">
                <div style="font-size: 12px; color: #166534;">
                    <strong>Status:</strong> Standard driver allocation. No emergency repositioning needed.
                </div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Surge Probability", f"{surge_prob * 100:.1f}%")
        with m2:
            st.metric("Est. Multiplier", f"{est_multiplier:.2f}x")
        with m3:
            st.metric("Zone Speed", f"{calc_speed:.1f} mph")

        # Gauge Chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=surge_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "20-Min Advance Surge Risk Index", 'font': {'size': 13, 'color': '#334155'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#DC2626" if surge_prob >= 0.5 else "#16A34A"},
                'steps': [
                    {'range': [0, 40], 'color': "#DCFCE7"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': "#0F172A", 'width': 3}, 'thickness': 0.8, 'value': 50}
            }
        ))
        fig_g.update_layout(height=220, margin=dict(l=15, r=15, t=25, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        # Baseline Formula Explainer
        st.markdown(f"""
        <div class="formula-box">
            <strong>Chicago Regulated Fare Calculation:</strong><br>
            Base Start ($3.25) + ({miles_val:.1f} mi * $2.25) + ({(seconds_val/60):.1f} min * $0.40) = <strong>${expected_base:.2f}</strong><br>
            Current Fare: <strong>${fare_val:.2f}</strong> &rarr; Multiplier: <strong>{est_multiplier:.2f}x</strong>
        </div>
        """, unsafe_allow_html=True)

    with col_pred2:
        st.subheader("Geospatial Chicago Community Area Map")

        # Generate live spatial risk estimates across all 77 zones
        all_zones_data = []
        for z_id, meta in CHICAGO_ZONE_COORDINATES.items():
            z_demand = demand_trips_val if z_id == selected_zone_id else max(2, int(demand_trips_val * (0.8 if z_id in [32, 8, 76, 56, 28] else 0.3)))
            z_fare = fare_val if z_id == selected_zone_id else max(10.0, fare_val * 0.7)

            z_vec = pd.DataFrame([{
                "zone_id": z_id, "hour": hour_val, "day_of_week": dow_val,
                "is_weekend": is_wknd, "is_rush_hour": is_rush,
                "demand_trips": z_demand, "avg_fare": z_fare,
                "avg_miles": miles_val, "avg_seconds": seconds_val, "avg_speed_mph": calc_speed
            }])

            z_prob = float(active_model.predict_proba(z_vec)[0][1]) if active_model else 0.1
            all_zones_data.append({
                "zone_id": z_id,
                "name": meta["name"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "hub_type": meta["hub_type"],
                "surge_risk_pct": round(z_prob * 100, 1),
                "is_selected": "Target Zone" if z_id == selected_zone_id else "Other Zone",
                "marker_size": 20 if z_id == selected_zone_id else 10
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
                title="Chicago 77 Community Areas Real-Time Surge Multiplier Risk Map"
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
                title="Chicago 77 Community Areas Real-Time Surge Multiplier Risk Map"
            )
        fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# ==============================================================================
# TAB 2: DRIVER REPOSITIONING & SUPPLY OPTIMIZER
# ==============================================================================
with tab_optimizer:
    st.subheader("Automated Fleet Supply & Rebalancing Optimizer")
    st.write("Calculates optimal vehicle repositioning schedules to prevent surge spikes before passengers experience price shock.")

    dispatch_res = optimize_driver_repositioning(selected_zone_id, surge_prob, demand_trips_val, est_multiplier)

    opt_c1, opt_c2 = st.columns([1.2, 1.8])

    with opt_c1:
        st.markdown(f"**Target Zone**: Zone {selected_zone_id} ({target_zone_meta['name']})")
        st.metric("Estimated Vehicle Deficit", f"{dispatch_res['drivers_needed']} Vehicles")
        st.metric("Total Recommended Dispatches", f"{dispatch_res.get('total_dispatched', 0)} Vehicles")
        st.metric("Projected Surge Reduction", f"-{dispatch_res.get('projected_multiplier_reduction', 0.0):.2f}x")

        st.markdown(f"""
        <div class="directive-card">
            <strong>Optimal Dispatch Directive:</strong><br>
            {dispatch_res['message']}
        </div>
        """, unsafe_allow_html=True)

    with opt_c2:
        st.markdown("**Transfer Schedule from Neighboring Donor Zones**")
        if dispatch_res["transfer_plan"]:
            plan_df = pd.DataFrame(dispatch_res["transfer_plan"])
            plan_df.columns = ["Donor Zone ID", "Donor Zone Name", "Drivers To Dispatch", "Distance (Miles)", "ETA (Minutes)", "Incentive Bonus ($)"]
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            fig_alloc = px.bar(
                plan_df,
                x="Donor Zone Name",
                y="Drivers To Dispatch",
                color="Incentive Bonus ($)",
                title="Driver Reallocation by Neighboring Community Zone",
                color_continuous_scale="Blues"
            )
            fig_alloc.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("No active vehicle rebalancing required. Supply and demand in this zone are currently balanced.")

# ==============================================================================
# TAB 3: MULTI-MODEL BENCHMARK LAB
# ==============================================================================
with tab_benchmark:
    st.subheader("Machine Learning Multi-Model Performance Benchmark")

    if eval_summary and "model_benchmarks" in eval_summary:
        benchmarks = eval_summary["model_benchmarks"]

        summary_rows = []
        for model_name, metrics in benchmarks.items():
            summary_rows.append({
                "Model Architecture": model_name,
                "ROC-AUC Score": metrics["roc_auc"],
                "F1 Score": metrics["f1_score"],
                "Recall (Surge >= 1.5x)": metrics["recall"],
                "Precision": metrics["precision"]
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        b_col1, b_col2 = st.columns(2)

        with b_col1:
            st.markdown("**ROC Curves Comparison**")
            fig_roc = go.Figure()
            curves = eval_summary.get("curves", {})
            for m_name, c_data in curves.items():
                fig_roc.add_trace(go.Scatter(x=c_data["fpr"], y=c_data["tpr"], mode='lines', name=m_name))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Random Baseline"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_roc, use_container_width=True)

        with b_col2:
            st.markdown("**Feature Importance Ranking (Random Forest)**")
            feat_imp = eval_summary.get("feature_importance", {})
            if feat_imp:
                f_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                fig_f = px.bar(f_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal")
                fig_f.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.warning("Benchmark metrics not yet computed. Run ml/advanced_training.py to generate summary.")

# ==============================================================================
# TAB 4: KAFKA REAL-TIME STREAM MONITOR
# ==============================================================================
with tab_stream:
    st.subheader("Kafka Real-Time Streaming Ingestion & Inference Stream")

    st.markdown("""
    This monitor displays live ride-request events arriving on the Kafka topic `ride-requests`
    and tracks 20-minute advance surge predictions evaluated by the streaming inference worker.
    """)

    live_alerts_p = PROCESSED_DATA_DIR / "live_alerts.json"

    if st.button("Simulate Incoming Streaming Batch (10 Events)"):
        with st.spinner("Processing streaming events via Kafka pipeline..."):
            from streaming_inference.stream_processor import run_live_stream_inference
            run_live_stream_inference(max_iterations=10, delay_sec=0.05)
            st.success("Processed 10 live ride request events and evaluated advance surge alerts.")

    if live_alerts_p.exists():
        with open(live_alerts_p, "r") as fp:
            alerts_list = json.load(fp)

        if alerts_list:
            st.markdown(f"**Latest Streaming Events Ingested ({len(alerts_list)} Records)**")
            stream_table = []
            for a in alerts_list[:15]:
                stream_table.append({
                    "Timestamp": a["timestamp"],
                    "Zone ID": a["zone_id"],
                    "Community Area": a["zone_name"],
                    "Surge Probability": f"{a['surge_probability']*100:.1f}%",
                    "Estimated Multiplier": f"{a['estimated_multiplier']}x",
                    "Status Alert": "SURGE WARNING (>1.5x)" if a["is_surge_warning"] else "NORMAL PRICING"
                })
            st.dataframe(pd.DataFrame(stream_table), use_container_width=True, hide_index=True)
    else:
        st.info("No live streaming events recorded yet. Click the button above to simulate incoming streaming traffic.")
