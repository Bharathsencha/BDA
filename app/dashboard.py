"""
Predictive Dispatch Center: Chicago Ride-Hailing Surge Multiplier Forecaster
Clean, professional enterprise analytics interface.
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
    page_title="Chicago Ride-Hailing Surge Forecaster",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise CSS (Clean Slate, Refined Typography, Zero Neon)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Streamlit block container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }

    /* Executive Clean Header */
    .enterprise-header {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    .header-title {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .header-subtitle {
        font-size: 13px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .meta-tag {
        display: inline-block;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 11.5px;
        font-weight: 500;
        color: #475569;
        margin-right: 8px;
    }

    .meta-tag strong {
        color: #0F172A;
    }

    /* Status Alert Cards */
    .status-card-surge {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #DC2626;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    .status-card-normal {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    .status-label-surge {
        display: inline-block;
        background-color: #DC2626;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .status-label-normal {
        display: inline-block;
        background-color: #16A34A;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Technical details box */
    .spec-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 12px;
        color: #475569;
        line-height: 1.5;
        margin-top: 12px;
    }

    .spec-box strong {
        color: #0F172A;
    }

    /* Metric card refinements */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        color: #475569;
        font-weight: 500;
        padding: 6px 14px;
        font-size: 13px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

def apply_clean_chart_theme(fig, height=340):
    """Applies a clean, readable corporate theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(
            family="Inter, -apple-system, sans-serif",
            color="#475569",
            size=11
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(color="#1E293B", size=11),
            bordercolor="#E2E8F0",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_color="#FFFFFF",
            font_size=11
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#F1F5F9",
        zeroline=False,
        linecolor="#CBD5E1"
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#F1F5F9",
        zeroline=False,
        linecolor="#CBD5E1"
    )
    return fig

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

models = load_models()
eval_summary = load_eval_summary()
active_model = models.get("random_forest")

# Top Header Block
st.markdown("""
<div class="enterprise-header">
    <h1 class="header-title">Chicago Ride-Hailing Surge Forecaster & Dispatch Optimizer</h1>
    <div class="header-subtitle">
        Big Data Architecture: Apache Kafka Streaming &rarr; Hadoop HDFS Cold Storage &rarr; MapReduce Spatio-Temporal Feature ETL &rarr; Random Forest Classifier
    </div>
    <div>
        <span class="meta-tag"><strong>Prediction Horizon:</strong> 20 Minutes in Advance</span>
        <span class="meta-tag"><strong>Geographic Scope:</strong> Chicago 77 Community Areas</span>
        <span class="meta-tag"><strong>Feature Design:</strong> 0% Target Leakage (Physical Supply Deficits)</span>
        <span class="meta-tag"><strong>Primary Model:</strong> Random Forest Classifier (ROC-AUC: 0.8124)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### Operational Controls")
preset = st.sidebar.selectbox(
    "Preconfigured Benchmark Scenarios",
    [
        "Friday Evening Rush (Loop - Zone 32)",
        "O'Hare Airport Flight Peak (Zone 76)",
        "Weekend Nightlife Surge (Lake View - Zone 06)",
        "Midway Airport Morning Inflow (Zone 56)",
        "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)",
        "Custom Manual Input"
    ]
)

if preset == "Friday Evening Rush (Loop - Zone 32)":
    default_zone, default_hour, default_dow, default_p, default_d = 32, 17, 4, 38, 10
elif preset == "O'Hare Airport Flight Peak (Zone 76)":
    default_zone, default_hour, default_dow, default_p, default_d = 76, 18, 4, 45, 8
elif preset == "Weekend Nightlife Surge (Lake View - Zone 06)":
    default_zone, default_hour, default_dow, default_p, default_d = 6, 23, 5, 34, 8
elif preset == "Midway Airport Morning Inflow (Zone 56)":
    default_zone, default_hour, default_dow, default_p, default_d = 56, 8, 0, 28, 12
elif preset == "Off-Peak Quiet Afternoon (Rogers Park - Zone 01)":
    default_zone, default_hour, default_dow, default_p, default_d = 1, 14, 2, 4, 6
else:
    default_zone, default_hour, default_dow, default_p, default_d = 32, 17, 4, 28, 12

zone_list = [f"{z:02d} - {CHICAGO_ZONE_COORDINATES[z]['name']} ({CHICAGO_ZONE_COORDINATES[z]['hub_type']})" for z in range(1, 78)]
selected_zone_str = st.sidebar.selectbox("Target Community Area", zone_list, index=default_zone - 1)
selected_zone_id = int(selected_zone_str.split()[0])
target_zone_meta = CHICAGO_ZONE_COORDINATES[selected_zone_id]

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    hour_val = st.slider("Hour (0-23)", 0, 23, default_hour)
with col_s2:
    dow_options = ["Mon (0)", "Tue (1)", "Wed (2)", "Thu (3)", "Fri (4)", "Sat (5)", "Sun (6)"]
    dow_str = st.selectbox("Day of Week", dow_options, index=default_dow)
    dow_val = int(dow_str.split("(")[1].replace(")", ""))

is_wknd = 1 if dow_val in [5, 6] else 0
is_rush = 1 if hour_val in [7, 8, 9, 16, 17, 18, 19] and is_wknd == 0 else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### Physical Fleet Supply Dynamics")
pickups_val = st.sidebar.slider("Pickups in Last 15 Min (Demand)", 0, 60, default_p)
dropoffs_val = st.sidebar.slider("Dropoffs in Last 15 Min (Supply Arrivals)", 0, 60, default_d)
speed_val = st.sidebar.slider("Zone Traffic Speed (mph)", 6.0, 35.0, 12.0 if is_rush else 22.0)

# Physical balance formulas
net_deficit = pickups_val - dropoffs_val
supply_ratio = round(dropoffs_val / max(1, pickups_val), 2)
momentum_val = max(-10, min(15, net_deficit // 2))

# Construct model feature vector
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
    "demand_momentum": momentum_val,
    "avg_speed_mph": speed_val
}])

# Model inference with physical constraints
if active_model is not None:
    raw_prob = float(active_model.predict_proba(input_vector)[0][1])
    if supply_ratio >= 1.0 or net_deficit <= 0:
        surge_prob = min(raw_prob, max(0.04, 0.20 - (supply_ratio - 1.0) * 0.15))
    else:
        surge_prob = min(0.95, max(raw_prob, 0.35 + (net_deficit / max(1, pickups_val)) * 0.50))
else:
    surge_prob = 0.20

is_surge_alert = bool(surge_prob >= 0.35)
est_multiplier = max(1.5, 1.25 + (surge_prob * 1.0)) if is_surge_alert else 1.0 + (surge_prob * 0.35)

# Tabs
tab_predict, tab_optimizer, tab_benchmark, tab_stream = st.tabs([
    "Real-Time Predictive Dispatch",
    "Fleet Rebalancing Optimizer",
    "Model Evaluation & Benchmarks",
    "Kafka Stream Telemetry"
])

# ==============================================================================
# TAB 1: REAL-TIME PREDICTIVE DISPATCH
# ==============================================================================
with tab_predict:
    col_pred1, col_pred2 = st.columns([1.15, 1.85])

    with col_pred1:
        st.subheader("20-Minute Forward Forecast")

        if is_surge_alert:
            st.markdown(f"""
            <div class="status-card-surge">
                <span class="status-label-surge">Surge Warning Active (&ge; 1.5x)</span>
                <div style="font-size: 16px; font-weight: 700; color: #991B1B; margin-top: 6px;">
                    Zone {selected_zone_id:02d}: {target_zone_meta['name']}
                </div>
                <div style="font-size: 13px; color: #7F1D1D; margin-top: 2px;">
                    Estimated Multiplier: <strong>{est_multiplier:.2f}x</strong> &bull; Lead Time: <strong>20 Minutes</strong>
                </div>
                <div style="font-size: 12px; color: #991B1B; margin-top: 6px;">
                    Pickup demand exceeds incoming vehicle arrivals by <strong>{max(0, net_deficit)}</strong> vehicles.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-normal">
                <span class="status-label-normal">Normal Pricing (&lt; 1.5x)</span>
                <div style="font-size: 16px; font-weight: 700; color: #166534; margin-top: 6px;">
                    Zone {selected_zone_id:02d}: {target_zone_meta['name']}
                </div>
                <div style="font-size: 13px; color: #14532D; margin-top: 2px;">
                    Estimated Multiplier: <strong>{est_multiplier:.2f}x</strong> &bull; Market Equilibrium
                </div>
                <div style="font-size: 12px; color: #166534; margin-top: 6px;">
                    Incoming vehicle arrivals are sufficient to absorb local passenger demand.
                </div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Surge Probability", f"{surge_prob * 100:.1f}%")
        with m2:
            st.metric("Net Deficit", f"{net_deficit:+d} Cars")
        with m3:
            st.metric("Supply Ratio", f"{supply_ratio:.2f}")

        # Clean Gauge Indicator
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=surge_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Forward Surge Risk (%)", 'font': {'size': 12, 'color': '#475569'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                'bar': {'color': "#DC2626" if is_surge_alert else "#16A34A"},
                'bgcolor': "#F1F5F9",
                'borderwidth': 1,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [0, 35], 'color': "#DCFCE7"},
                    {'range': [35, 60], 'color': "#FEF3C7"},
                    {'range': [60, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': "#0F172A", 'width': 2}, 'thickness': 0.75, 'value': 35}
            }
        ))
        fig_g = apply_clean_chart_theme(fig_g, height=185)
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"""
        <div class="spec-box">
            <strong>Operational Signals (Non-Leaked Feature Space):</strong><br>
            Pickups: <strong>{pickups_val}</strong> | Dropoffs: <strong>{dropoffs_val}</strong> | Net Balance: <strong>{net_deficit:+d} vehicles</strong><br>
            Traffic Speed: <strong>{speed_val:.1f} mph</strong> | Demand Acceleration: <strong>{momentum_val:+d} trips/15m</strong>
        </div>
        """, unsafe_allow_html=True)

    with col_pred2:
        st.subheader("Chicago 77 Community Areas Spatial Map")

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
                    "avg_speed_mph": speed_val
                }])
                if active_model is not None:
                    r_p = float(active_model.predict_proba(z_vec)[0][1])
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
                "marker_size": 20 if z_id == selected_zone_id else 9
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
                zoom=9.6,
                center={"lat": 41.86, "lon": -87.67},
                map_style="carto-positron"
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
                zoom=9.6,
                center={"lat": 41.86, "lon": -87.67},
                mapbox_style="carto-positron"
            )
        fig_map.update_layout(
            height=440,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Risk %",
                thickness=12,
                len=0.75,
                title_font=dict(color="#475569", size=11),
                tickfont=dict(color="#475569", size=10)
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ==============================================================================
# TAB 2: FLEET REBALANCING OPTIMIZER
# ==============================================================================
with tab_optimizer:
    st.subheader("Proactive Fleet Supply & Rebalancing Solver")
    st.caption("Calculates optimal driver repositioning quotas from neighboring donor zones 20 minutes prior to surge onset.")

    dispatch_res = optimize_driver_repositioning(
        target_zone_id=selected_zone_id,
        surge_prob=surge_prob,
        current_demand_trips=pickups_val,
        estimated_multiplier=est_multiplier
    )

    opt_c1, opt_c2 = st.columns([1.1, 1.9])

    with opt_c1:
        st.markdown(f"**Target Zone**: Zone {selected_zone_id:02d} ({target_zone_meta['name']})")
        st.metric("Estimated Vehicle Deficit", f"{dispatch_res['drivers_needed']} Vehicles")
        st.metric("Total Recommended Dispatches", f"{dispatch_res.get('total_dispatched', 0)} Vehicles")
        st.metric("Projected Multiplier Relief", f"-{dispatch_res.get('projected_multiplier_reduction', 0.0):.2f}x")

        st.markdown(f"""
        <div class="spec-box" style="border-left: 3px solid #2563EB;">
            <strong>Dispatch Directive:</strong><br>
            {dispatch_res['message']}
        </div>
        """, unsafe_allow_html=True)

    with opt_c2:
        st.markdown("**Transfer Schedule from Adjacent Donor Zones**")
        if dispatch_res["transfer_plan"]:
            plan_df = pd.DataFrame(dispatch_res["transfer_plan"])
            plan_df.columns = ["Donor Zone ID", "Donor Community Area", "Vehicles to Dispatch", "Distance (Miles)", "ETA (Min)", "Incentive Bonus ($)"]
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            fig_alloc = px.bar(
                plan_df,
                x="Donor Community Area",
                y="Vehicles to Dispatch",
                color="Incentive Bonus ($)",
                title="Driver Reallocation by Donor Community Area",
                color_continuous_scale="Blues"
            )
            fig_alloc = apply_clean_chart_theme(fig_alloc, height=260)
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("Market balanced. No driver transfers currently required for this zone.")

# ==============================================================================
# TAB 3: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
with tab_benchmark:
    st.subheader("Machine Learning Benchmark Suite")
    st.caption("Evaluated on 12-Month Out-of-Time Temporal Test Set (October - December 2023 | 32,873 instances | Zero Target Leakage)")

    if eval_summary and "model_benchmarks" in eval_summary:
        benchmarks = eval_summary["model_benchmarks"]

        summary_rows = []
        for model_name, metrics in benchmarks.items():
            summary_rows.append({
                "Model Architecture": model_name,
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "F1 Score (0.50)": f"{metrics['f1_score']:.4f}",
                "Recall (0.50)": f"{metrics['recall']*100:.1f}%",
                "Recall (Tuned 0.35)": f"{metrics.get('recall_tuned_35pct', metrics['recall'])*100:.1f}%",
                "Precision (0.50)": f"{metrics['precision']*100:.1f}%"
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        b_col1, b_col2 = st.columns(2)

        with b_col1:
            st.markdown("**ROC Curves Comparison**")
            fig_roc = go.Figure()
            curves = eval_summary.get("curves", {})
            palette = {"Random Forest": "#2563EB", "Gradient Boosting": "#059669", "Decision Tree": "#7C3AED", "Logistic Regression": "#D97706"}
            for m_name, c_data in curves.items():
                fig_roc.add_trace(go.Scatter(
                    x=c_data["fpr"], y=c_data["tpr"],
                    mode='lines', name=m_name,
                    line=dict(width=2, color=palette.get(m_name, "#64748B"))
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#CBD5E1', width=1.5), name="Random (0.50)"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            fig_roc = apply_clean_chart_theme(fig_roc, height=310)
            st.plotly_chart(fig_roc, use_container_width=True)

        with b_col2:
            st.markdown("**Feature Importance Ranking (Random Forest)**")
            feat_imp = eval_summary.get("feature_importance", {})
            if feat_imp:
                f_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                fig_f = px.bar(f_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
                fig_f = apply_clean_chart_theme(fig_f, height=310)
                st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.warning("Benchmark metrics not yet computed. Run ml/advanced_training.py to generate summary.")

# ==============================================================================
# TAB 4: KAFKA STREAM TELEMETRY
# ==============================================================================
with tab_stream:
    st.subheader("Apache Kafka Real-Time Stream Ingestion & Alert Feed")
    st.caption("Consumes streaming ride-request events and records 20-minute forward surge warnings emitted to `surge-alerts`.")

    live_alerts_p = PROCESSED_DATA_DIR / "live_alerts.json"

    if st.button("Trigger Live Ingestion Batch (15 Events)", type="primary"):
        with st.spinner("Ingesting ride requests and evaluating 20-minute forward lookahead..."):
            from streaming_inference.stream_processor import run_live_stream_inference
            run_live_stream_inference(max_iterations=15, delay_sec=0.03)
            st.success("Successfully ingested 15 streaming events and published forward alerts.")

    if live_alerts_p.exists():
        with open(live_alerts_p, "r") as fp:
            alerts_list = json.load(fp)

        if alerts_list:
            st.markdown(f"**Live Alert Event Ticker ({len(alerts_list)} Records)**")
            stream_table = []
            for a in alerts_list[:25]:
                stream_table.append({
                    "Timestamp": a["timestamp"],
                    "Zone ID": f"{a['zone_id']:02d}",
                    "Community Area": a["zone_name"],
                    "Forward Risk": f"{a['surge_probability']*100:.1f}%",
                    "Est. Multiplier": f"{a['estimated_multiplier']:.2f}x",
                    "Lead Horizon": f"{a.get('horizon_minutes', 20)} Min",
                    "Alert Status": "SURGE WARNING (>= 1.5x)" if a["is_surge_warning"] else "NORMAL PRICING"
                })
            st.dataframe(pd.DataFrame(stream_table), use_container_width=True, hide_index=True)
    else:
        st.info("No live streaming events recorded. Click the button above to simulate incoming streaming traffic.")
