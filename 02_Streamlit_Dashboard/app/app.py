"""
Streamlit Web Application — Manufacturing Efficiency Classification & Validation
Portfolio Case Study | Sensor, Production & Network Telemetry
================================================================
Run with: streamlit run app.py

Methodological note
-------------------
Efficiency_Status is almost deterministically reconstructable from contemporaneous
Error_Rate_% and Production_Speed_units_per_hr. The transparent business rule is
therefore the primary current-state operational method. Machine-learning models are
retained as benchmark models. Temporal diagnostics did not support a defensible
future-risk prediction claim for this dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Keep classic object-backed string behavior for Streamlit/pandas stability.
pd.set_option("future.infer_string", False)

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Manufacturing Efficiency Analytics | Portfolio Case Study",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "manufacturing_features.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
VALIDATION_DIR = OUTPUTS_DIR / "validation"

STATUS_COLORS = {
    "High": "#2ca02c",
    "Medium": "#ff9f1c",
    "Low": "#d62728",
}
STATUS_ORDER = ["Low", "Medium", "High"]

FEATURE_GROUPS = {
    "Network": [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Network_Reliability_Score",
    ],
    "Sensor / Maintenance": [
        "Temperature_C",
        "Vibration_Hz",
        "Sensor_Stability_Score",
        "Predictive_Maintenance_Score",
    ],
    "Production / Quality": [
        "Production_Speed_units_per_hr",
        "Error_Rate_%",
        "Quality_Control_Defect_Rate_%",
        "Error_to_Output_Ratio",
        "Quality_Adjusted_Error",
        "Power_Consumption_kW",
        "Energy_Efficiency_Ratio",
    ],
}


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Machine_ID"] = df["Machine_ID"].astype("int64")
    for column in ["Operation_Mode", "Efficiency_Status"]:
        df[column] = df[column].astype("object")
    return df


@st.cache_resource
def load_models():
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgboost_model.json")

    models = {
        "XGBoost": xgb_model,
        "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.joblib"),
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression_model.joblib"
        ),
    }

    # Conservative inference settings for hosted environments.
    try:
        models["XGBoost"].set_params(n_jobs=1)
    except Exception:
        pass
    try:
        models["Random Forest"].n_jobs = 1
    except Exception:
        pass

    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

    metadata = json.loads(
        (MODEL_DIR / "model_meta.json").read_text(encoding="utf-8")
    )

    return models, scaler, label_encoder, metadata


@st.cache_data
def load_model_results() -> dict:
    return json.loads(
        (OUTPUTS_DIR / "model_results.json").read_text(encoding="utf-8")
    )


@st.cache_data
def load_importance():
    def load_series(path: Path) -> pd.Series:
        if not path.exists():
            return pd.Series(dtype=float)
        frame = pd.read_csv(path, index_col=0)
        if frame.empty:
            return pd.Series(dtype=float)
        return frame.iloc[:, 0]

    return (
        load_series(OUTPUTS_DIR / "rf_feature_importance.csv"),
        load_series(OUTPUTS_DIR / "xgb_feature_importance.csv"),
        load_series(OUTPUTS_DIR / "shap_global_importance.csv"),
    )


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Core classification helpers
# ------------------------------------------------------------------
def business_rule_predict(frame: pd.DataFrame) -> np.ndarray:
    """Transparent current-state efficiency rule discovered in diagnostics."""
    error = frame["Error_Rate_%"].to_numpy()
    speed = frame["Production_Speed_units_per_hr"].to_numpy()

    predictions = np.full(len(frame), "Medium", dtype=object)

    low_mask = (error > 5) | (speed < 200)
    high_mask = (error <= 2) & (speed >= 400) & (~low_mask)

    predictions[low_mask] = "Low"
    predictions[high_mask] = "High"
    return predictions


def encode_for_models(frame: pd.DataFrame, mode_columns: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(
        frame,
        columns=["Operation_Mode"],
        prefix="Mode",
        dtype=int,
    )
    for column in mode_columns:
        if column not in encoded.columns:
            encoded[column] = 0
    return encoded


def predict_row(method_name: str, row_df: pd.DataFrame):
    """Current-state classification. Returns (label, probabilities-or-None)."""
    if method_name == "Transparent Business Rule":
        return business_rule_predict(row_df)[0], None

    model = models[method_name]
    X = row_df[feature_cols]

    if method_name == "Logistic Regression":
        X = X.copy()
        X[numeric_features] = scaler.transform(X[numeric_features])

    probabilities = model.predict_proba(X)[0]
    pred_index = int(np.argmax(probabilities))
    pred_label = le.inverse_transform([pred_index])[0]
    probability_dict = {
        le.inverse_transform([index])[0]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    return pred_label, probability_dict


def batch_classify(method_name: str, batch: pd.DataFrame) -> pd.DataFrame:
    result = batch.copy()

    if method_name == "Transparent Business Rule":
        result["Predicted"] = business_rule_predict(result)
        result["Max_Class_Probability"] = np.nan
        return result

    encoded = encode_for_models(result, mode_columns)
    X = encoded[feature_cols]

    if method_name == "Logistic Regression":
        X = X.copy()
        X[numeric_features] = scaler.transform(X[numeric_features])

    probabilities = models[method_name].predict_proba(X)
    result["Predicted"] = le.inverse_transform(np.argmax(probabilities, axis=1))
    result["Max_Class_Probability"] = probabilities.max(axis=1) * 100
    return result


# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
try:
    df = load_data()
    models, scaler, le, meta = load_models()
    model_results = load_model_results()
    rf_imp, xgb_imp, shap_imp = load_importance()
except Exception as exc:
    st.error(
        "**Could not load the project artifacts.**\n\n"
        "Run the rebuilt modeling/explainability pipeline first and make sure "
        "the pinned dependencies are installed.\n\n"
        f"Underlying error: `{type(exc).__name__}: {exc}`"
    )
    st.stop()

# New rebuilt metadata contract.
feature_cols = meta["feature_columns"]
numeric_features = meta["numeric_features"]
class_names = meta["class_names"]
mode_columns = meta["mode_columns"]

# New rebuilt result contract.
ml_model_results = model_results.get("models", {})
baseline_results = model_results.get("baselines", {})
methodology = model_results.get("methodology", {})

# Validation artifacts are optional so the app remains deployable even if a
# user has not rerun every diagnostic locally yet.
temporal_summary = load_optional_csv(METRICS_DIR / "temporal_model_summary.csv")
holdout_results = load_optional_csv(METRICS_DIR / "holdout_model_results.csv")
ablation_results = load_optional_csv(METRICS_DIR / "feature_ablation_results.csv")
final_comparison = load_optional_csv(METRICS_DIR / "final_model_comparison.csv")
target_diagnostics = load_optional_json(
    DIAGNOSTICS_DIR / "target_diagnostics_summary.json"
)
temporal_diagnostics = load_optional_json(
    DIAGNOSTICS_DIR / "temporal_signal_diagnostics.json"
)


# ------------------------------------------------------------------
# Sidebar — filters and classification method
# ------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Manufacturing Analytics")
st.sidebar.caption("Manufacturing Efficiency Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

machine_list = sorted(df["Machine_ID"].unique().tolist())
selected_machines = st.sidebar.multiselect(
    "Machine selector",
    options=machine_list,
    default=machine_list,
    help="Choose one or more machines.",
)

mode_list = sorted(df["Operation_Mode"].unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Operation mode",
    options=mode_list,
    default=mode_list,
)

min_date = df["Datetime"].min()
max_date = df["Datetime"].max()
date_range = st.sidebar.slider(
    "Time window filter",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="DD-MM-YY HH:mm",
)

network_range = st.sidebar.slider(
    "Network reliability score",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0,
    help="0 = worst network reliability, 100 = best.",
)

st.sidebar.markdown("### 🎚️ Operational Thresholds")
error_sensitivity = st.sidebar.slider(
    "Error Rate threshold (%)", 0.0, 15.0, 5.0, 0.5
)
speed_sensitivity = st.sidebar.slider(
    "Low production-speed threshold (units/hr)", 0.0, 500.0, 200.0, 10.0
)

st.sidebar.markdown("---")
decision_methods = [
    "Transparent Business Rule",
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]
method_choice = st.sidebar.selectbox(
    "Current-state classification method",
    options=decision_methods,
    index=0,
    help=(
        "The transparent rule is the primary operational method. "
        "ML models are retained as current-state benchmark models."
    ),
)

if not selected_machines:
    selected_machines = machine_list
if not selected_modes:
    selected_modes = mode_list

mask = (
    df["Machine_ID"].isin(selected_machines)
    & df["Operation_Mode"].isin(selected_modes)
    & (df["Datetime"] >= pd.Timestamp(date_range[0]))
    & (df["Datetime"] <= pd.Timestamp(date_range[1]))
    & (df["Network_Reliability_Score"] >= network_range[0])
    & (df["Network_Reliability_Score"] <= network_range[1])
)
fdf = df.loc[mask].copy()
st.sidebar.markdown(f"**Filtered records:** {len(fdf):,} / {len(df):,}")


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("⚙️ Manufacturing Efficiency Classification & Validation")
st.caption("Sensor, production & network telemetry · Thales Group smart-manufacturing portfolio case study")

st.info(
    "**Methodology note:** this dashboard performs current-state efficiency "
    "classification. Target diagnostics show that `Efficiency_Status` is almost "
    "entirely defined by contemporaneous error rate and production speed. ML "
    "models are therefore presented as benchmark models, not as validated "
    "future-risk predictors."
)

if fdf.empty:
    st.warning("No records match the current filters. Please broaden your selection.")
    st.stop()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📡 Current-State Classification",
        "🏭 Machine Insights",
        "🔬 Explainability",
        "🌐 Operational Monitoring",
        "✅ Model Validation",
    ]
)


# ==================================================================
# TAB 1 — Current-state classification
# ==================================================================
with tab1:
    st.subheader("Current-State Efficiency Classification")

    class_counts = fdf["Efficiency_Status"].value_counts()
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Records in view", f"{len(fdf):,}")
    kc2.metric(
        "% High",
        f"{class_counts.get('High', 0) / len(fdf) * 100:.1f}%",
    )
    kc3.metric(
        "% Medium",
        f"{class_counts.get('Medium', 0) / len(fdf) * 100:.1f}%",
    )
    kc4.metric(
        "% Low",
        f"{class_counts.get('Low', 0) / len(fdf) * 100:.1f}%",
    )

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("#### 🎛️ Simulate a Machine Reading")
        st.caption(
            "Adjust contemporaneous machine values and classify the current efficiency state."
        )

        c1, c2 = st.columns(2)
        with c1:
            temp = st.slider(
                "Temperature (°C)", 30.0, 90.0, float(fdf["Temperature_C"].mean())
            )
            vib = st.slider(
                "Vibration (Hz)", 0.0, 5.0, float(fdf["Vibration_Hz"].mean())
            )
            power = st.slider(
                "Power Consumption (kW)",
                0.0,
                10.0,
                float(fdf["Power_Consumption_kW"].mean()),
            )
            latency = st.slider(
                "Network Latency (ms)",
                0.0,
                50.0,
                float(fdf["Network_Latency_ms"].mean()),
            )
            packet_loss = st.slider(
                "Packet Loss (%)",
                0.0,
                5.0,
                float(fdf["Packet_Loss_%"].mean()),
            )
        with c2:
            defect = st.slider(
                "QC Defect Rate (%)",
                0.0,
                10.0,
                float(fdf["Quality_Control_Defect_Rate_%"].mean()),
            )
            speed = st.slider(
                "Production Speed (units/hr)",
                0.0,
                500.0,
                float(fdf["Production_Speed_units_per_hr"].mean()),
            )
            maint = st.slider(
                "Predictive Maintenance Score",
                0.0,
                1.0,
                float(fdf["Predictive_Maintenance_Score"].mean()),
            )
            error = st.slider(
                "Error Rate (%)", 0.0, 15.0, float(fdf["Error_Rate_%"].mean())
            )
            op_mode = st.selectbox("Operation Mode", options=mode_list)

        if st.button("Classify Current Efficiency", type="primary", width="stretch"):
            # The feature row contains both numeric values and the categorical
            # operation mode; keep the mapping value type broad enough for all
            # of the assignments below.
            row: dict[str, object] = {column: 0 for column in feature_cols}
            row.update(
                {
                    "Temperature_C": temp,
                    "Vibration_Hz": vib,
                    "Power_Consumption_kW": power,
                    "Network_Latency_ms": latency,
                    "Packet_Loss_%": packet_loss,
                    "Quality_Control_Defect_Rate_%": defect,
                    "Production_Speed_units_per_hr": speed,
                    "Predictive_Maintenance_Score": maint,
                    "Error_Rate_%": error,
                    "Hour": 12,
                    "Is_Weekend": 0,
                    "Operation_Mode": op_mode,
                }
            )

            row["Sensor_Stability_Score"] = float(
                fdf["Sensor_Stability_Score"].mean()
            )
            row["Energy_Efficiency_Ratio"] = speed / power if power > 0 else 0
            row["Error_to_Output_Ratio"] = (
                error / speed * 1000 if speed > 0 else 0
            )
            row["Quality_Adjusted_Error"] = error + defect

            lat_min = df["Network_Latency_ms"].min()
            lat_max = df["Network_Latency_ms"].max()
            loss_min = df["Packet_Loss_%"].min()
            loss_max = df["Packet_Loss_%"].max()
            lat_norm = (latency - lat_min) / (lat_max - lat_min)
            loss_norm = (packet_loss - loss_min) / (loss_max - loss_min)
            row["Network_Reliability_Score"] = (
                1 - (0.5 * lat_norm + 0.5 * loss_norm)
            ) * 100

            row_df = pd.DataFrame([row])
            if method_choice != "Transparent Business Rule":
                row_df = encode_for_models(row_df, mode_columns)

            pred_label, proba_dict = predict_row(method_choice, row_df)
            st.session_state["last_pred"] = (
                method_choice,
                pred_label,
                proba_dict,
                error,
                speed,
            )

        if "last_pred" in st.session_state:
            used_method, pred_label, proba_dict, used_error, used_speed = st.session_state[
                "last_pred"
            ]

            st.markdown(
                f"### Classified Efficiency: "
                f"<span style='color:{STATUS_COLORS[pred_label]}; font-size:1.5em;'>●</span> "
                f"**{pred_label}**",
                unsafe_allow_html=True,
            )
            st.caption(f"Method: **{used_method}**")

            if proba_dict is None:
                st.caption(
                    "The transparent rule is deterministic, so a probability/confidence score is not applicable."
                )
                if pred_label == "Low":
                    st.warning(
                        "Low status is triggered when Error Rate > 5% or Production Speed < 200 units/hr."
                    )
                elif pred_label == "High":
                    st.success(
                        "High status requires Error Rate ≤ 2% and Production Speed ≥ 400 units/hr."
                    )
                else:
                    st.info(
                        "Medium status applies when neither the Low nor High rule is satisfied."
                    )
            else:
                probability_fig = go.Figure(
                    go.Bar(
                        x=[proba_dict.get(status, 0) * 100 for status in STATUS_ORDER],
                        y=STATUS_ORDER,
                        orientation="h",
                        marker_color=[STATUS_COLORS[status] for status in STATUS_ORDER],
                        text=[
                            f"{proba_dict.get(status, 0) * 100:.1f}%"
                            for status in STATUS_ORDER
                        ],
                        textposition="outside",
                    )
                )
                probability_fig.update_layout(
                    title="ML Class Probability Distribution",
                    xaxis_title="Predicted probability (%)",
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_range=[0, 110],
                )
                st.plotly_chart(probability_fig, width="stretch")
                st.caption(
                    "These are current-state benchmark-model probabilities, not probabilities of future degradation."
                )

    with right:
        st.markdown("#### Filtered-Batch Agreement")
        st.caption(
            f"Method **{method_choice}** applied to a sample of the filtered dataset."
        )

        batch = fdf.sample(n=min(1500, len(fdf)), random_state=1).copy()
        batch_scored = batch_classify(method_choice, batch)
        agreement = (
            batch_scored["Predicted"] == batch_scored["Efficiency_Status"]
        ).mean() * 100
        st.metric("Agreement with logged status", f"{agreement:.1f}%")

        if method_choice != "Transparent Business Rule":
            fig = px.histogram(
                batch_scored,
                x="Max_Class_Probability",
                color="Predicted",
                nbins=25,
                color_discrete_map=STATUS_COLORS,
                title="Maximum ML Class Probability",
            )
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Probability concentration is shown for benchmark diagnostics only."
            )
        else:
            rule_counts = (
                batch_scored["Predicted"].value_counts().reindex(STATUS_ORDER, fill_value=0)
            )
            fig = px.bar(
                x=rule_counts.index,
                y=rule_counts.values,
                color=rule_counts.index,
                color_discrete_map=STATUS_COLORS,
                title="Rule Classification Distribution",
                labels={"x": "Efficiency Status", "y": "Records"},
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("##### Recent Classifications")
        show_cols = [
            "Datetime",
            "Machine_ID",
            "Operation_Mode",
            "Efficiency_Status",
            "Predicted",
        ]
        if method_choice != "Transparent Business Rule":
            show_cols.append("Max_Class_Probability")

        display_batch = batch_scored.sort_values("Datetime", ascending=False)[show_cols].head(15)
        if "Max_Class_Probability" in display_batch.columns:
            st.dataframe(
                display_batch.style.format({"Max_Class_Probability": "{:.1f}%"}),
                width="stretch",
                height=320,
            )
        else:
            st.dataframe(display_batch, width="stretch", height=320)


# ==================================================================
# TAB 2 — Machine-level descriptive analytics
# ==================================================================
with tab2:
    st.subheader("Machine-Level Efficiency Patterns")
    st.caption(
        "These views are descriptive summaries of logged records; they are not forecasts."
    )

    default_focus = selected_machines[:5] if len(selected_machines) > 5 else selected_machines
    machine_focus = st.multiselect(
        "Focus machines (leave empty = all filtered machines)",
        options=selected_machines,
        default=default_focus,
    )
    focus_df = fdf[fdf["Machine_ID"].isin(machine_focus)] if machine_focus else fdf

    col1, col2 = st.columns([1.3, 1])
    with col1:
        trend = (
            focus_df.groupby([focus_df["Datetime"].dt.date, "Machine_ID"])[
                "Efficiency_Status"
            ]
            .apply(lambda series: (series == "Low").mean() * 100)
            .reset_index(name="Pct_Low")
        )
        trend.columns = ["Date", "Machine_ID", "Pct_Low"]
        trend["Machine_ID"] = trend["Machine_ID"].astype(str)
        fig = px.line(
            trend,
            x="Date",
            y="Pct_Low",
            color="Machine_ID",
            title="% Low-Efficiency Records per Day, by Machine",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    with col2:
        rank = (
            fdf.groupby("Machine_ID")["Efficiency_Status"]
            .apply(lambda series: (series == "Low").mean() * 100)
            .sort_values(ascending=False)
            .reset_index(name="Pct_Low")
        )
        rank["Machine_ID"] = rank["Machine_ID"].astype(str)
        fig = px.bar(
            rank,
            x="Pct_Low",
            y="Machine_ID",
            orientation="h",
            title="Share of Low Records by Machine",
            height=420,
            color="Pct_Low",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Machine × Efficiency Status")
    pattern = pd.crosstab(fdf["Machine_ID"], fdf["Efficiency_Status"])
    for status in STATUS_ORDER:
        if status not in pattern.columns:
            pattern[status] = 0
    pattern = pattern[STATUS_ORDER]
    pattern_pct = pattern.div(pattern.sum(axis=1), axis=0) * 100
    fig = px.imshow(
        pattern_pct.T,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Machine ID", y="Efficiency Status", color="% of records"),
        title="Historical Efficiency Classification Heatmap",
    )
    st.plotly_chart(fig, width="stretch")

    summary_tbl = (
        fdf.groupby("Machine_ID")
        .agg(
            Records=("Efficiency_Status", "count"),
            Pct_Low=("Efficiency_Status", lambda series: (series == "Low").mean() * 100),
            Pct_Medium=(
                "Efficiency_Status", lambda series: (series == "Medium").mean() * 100
            ),
            Pct_High=("Efficiency_Status", lambda series: (series == "High").mean() * 100),
            Avg_Error_Rate=("Error_Rate_%", "mean"),
            Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
            Avg_Maintenance_Score=("Predictive_Maintenance_Score", "mean"),
        )
        .round(2)
        .sort_values("Pct_Low", ascending=False)
    )
    st.markdown("##### Machine Summary")
    st.dataframe(summary_tbl, width="stretch", height=350)


# ==================================================================
# TAB 3 — Explainability
# ==================================================================
with tab3:
    st.subheader("ML Benchmark Explainability")
    st.warning(
        "Feature importance and SHAP explain how the ML benchmark reconstructs the current label. "
        "They should not be interpreted as causal drivers or future-risk evidence."
    )

    c1, c2 = st.columns(2)
    with c1:
        if shap_imp.empty:
            st.info("Run `04_explainability.py` to generate SHAP artifacts.")
        else:
            shap_top = shap_imp.sort_values(ascending=False).head(12)
            fig = px.bar(
                shap_top[::-1],
                orientation="h",
                title="Global SHAP Importance",
                labels={"value": "mean |SHAP value|", "index": "Feature"},
            )
            fig.update_layout(showlegend=False, height=430)
            st.plotly_chart(fig, width="stretch")

    with c2:
        model_imp_choice = st.radio(
            "Importance source", ["Random Forest", "XGBoost"], horizontal=True
        )
        imp_series = rf_imp if model_imp_choice == "Random Forest" else xgb_imp
        if imp_series.empty:
            st.info("Feature-importance artifact is not available.")
        else:
            top = imp_series.sort_values(ascending=False).head(12)
            fig = px.bar(
                top[::-1],
                orientation="h",
                title=f"{model_imp_choice} Feature Importance",
                labels={"value": "Importance", "index": "Feature"},
            )
            fig.update_layout(showlegend=False, height=430)
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("##### Example Current-State Record")
    sample_for_explain = fdf.sample(n=min(300, len(fdf)), random_state=7).reset_index()
    sample_for_explain["label"] = (
        sample_for_explain["Datetime"].astype(str)
        + " | Machine "
        + sample_for_explain["Machine_ID"].astype(str)
        + " | "
        + sample_for_explain["Efficiency_Status"]
    )
    choice = st.selectbox("Select a record", options=sample_for_explain["label"])
    row = sample_for_explain[sample_for_explain["label"] == choice].iloc[0]

    raw_row = pd.DataFrame([row]).drop(columns=["label"], errors="ignore")
    encoded_row = encode_for_models(raw_row, mode_columns)
    xgb_label, xgb_probabilities = predict_row("XGBoost", encoded_row)
    rule_label, _ = predict_row("Transparent Business Rule", raw_row)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Logged status", str(row["Efficiency_Status"]))
    m2.metric("Rule classification", rule_label)
    m3.metric("XGBoost benchmark", xgb_label)
    m4.metric(
        "Max XGBoost class probability",
        (
            f"{max(xgb_probabilities.values()) * 100:.1f}%"
            if xgb_probabilities
            else "N/A"
        ),
    )

    explain_cols = [
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
        "Error_to_Output_Ratio",
        "Quality_Adjusted_Error",
        "Network_Reliability_Score",
        "Sensor_Stability_Score",
    ]
    comparison = pd.DataFrame(
        {
            "Metric": explain_cols,
            "This Record": [row[column] for column in explain_cols],
            "Dataset Average": [df[column].mean() for column in explain_cols],
        }
    )
    comparison["This Record"] = comparison["This Record"].round(2)
    comparison["Dataset Average"] = comparison["Dataset Average"].round(2)
    st.dataframe(comparison, width="stretch", hide_index=True)

    error_rate = float(str(row["Error_Rate_%"]))
    production_speed = float(str(row["Production_Speed_units_per_hr"]))

    if error_rate > 5:
        st.error("Rule condition triggered: Error Rate > 5%.")
    elif production_speed < 200:
        st.error("Rule condition triggered: Production Speed < 200 units/hr.")
    elif error_rate <= 2 and production_speed >= 400:
        st.success("High-rule conditions are satisfied.")
    else:
        st.info("The record falls into the Medium rule region.")


# ==================================================================
# TAB 4 — Operational monitoring
# ==================================================================
with tab4:
    st.subheader("Operational Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        operation_mix = (
            pd.crosstab(
                fdf["Operation_Mode"],
                fdf["Efficiency_Status"],
                normalize="index",
            )
            * 100
        )
        for status in STATUS_ORDER:
            if status not in operation_mix.columns:
                operation_mix[status] = 0
        operation_mix = (
            operation_mix[STATUS_ORDER]
            .reset_index()
            .melt(
                id_vars="Operation_Mode",
                var_name="Efficiency_Status",
                value_name="Pct",
            )
        )
        fig = px.bar(
            operation_mix,
            x="Operation_Mode",
            y="Pct",
            color="Efficiency_Status",
            barmode="stack",
            color_discrete_map=STATUS_COLORS,
            title="Efficiency Mix by Operation Mode (%)",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        if shap_imp.empty:
            st.info("Run explainability analysis to generate grouped SHAP impact.")
        else:
            group_scores = {}
            for group, features in FEATURE_GROUPS.items():
                valid_features = [feature for feature in features if feature in shap_imp.index]
                group_scores[group] = float(shap_imp[valid_features].sum())
            grouped = pd.DataFrame(
                {
                    "Group": list(group_scores.keys()),
                    "Total |SHAP| Impact": list(group_scores.values()),
                }
            )
            fig = px.pie(
                grouped,
                names="Group",
                values="Total |SHAP| Impact",
                hole=0.45,
                title="Relative Benchmark-Model Impact",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Production/quality variables dominate because they encode the current target rule."
            )

    st.markdown("---")
    fig = px.box(
        fdf,
        x="Efficiency_Status",
        y="Network_Reliability_Score",
        color="Efficiency_Status",
        category_orders={"Efficiency_Status": STATUS_ORDER},
        color_discrete_map=STATUS_COLORS,
        title="Network Reliability by Logged Efficiency Class",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Threshold-Flagged Current Records")
    flagged = fdf[
        (fdf["Error_Rate_%"] > error_sensitivity)
        | (fdf["Production_Speed_units_per_hr"] < speed_sensitivity)
    ]
    st.metric(
        "Records meeting selected thresholds",
        f"{len(flagged):,} ({len(flagged) / len(fdf) * 100:.1f}% of filtered data)",
    )
    st.caption(
        "These are current threshold flags, not predicted future failures."
    )
    st.dataframe(
        flagged.sort_values("Datetime", ascending=False)[
            [
                "Datetime",
                "Machine_ID",
                "Operation_Mode",
                "Error_Rate_%",
                "Production_Speed_units_per_hr",
                "Efficiency_Status",
            ]
        ].head(20),
        width="stretch",
    )

    st.markdown("---")
    st.subheader("Current-State Benchmark Summary")

    baseline_rows = []
    for name, result in baseline_results.items():
        baseline_rows.append(
            {
                "Method": name,
                "Accuracy": result.get("accuracy"),
                "Balanced Accuracy": result.get("balanced_accuracy"),
                "Macro F1": result.get("macro_f1"),
                "Weighted F1": result.get("weighted_f1"),
            }
        )
    st.markdown("##### Baselines")
    st.dataframe(
        pd.DataFrame(baseline_rows).round(4),
        width="stretch",
        hide_index=True,
    )

    model_rows = []
    for name, result in ml_model_results.items():
        model_rows.append(
            {
                "Model": name,
                "Accuracy": result.get("accuracy"),
                "Balanced Accuracy": result.get("balanced_accuracy"),
                "Macro F1": result.get("macro_f1"),
                "Weighted F1": result.get("weighted_f1"),
                "Macro Avg Precision": result.get("average_precision_macro"),
                "Calibration Error": result.get("expected_calibration_error"),
            }
        )
    st.markdown("##### ML Benchmarks")
    st.dataframe(
        pd.DataFrame(model_rows).round(4),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "High RF/XGBoost scores measure reconstruction of the contemporaneous label rule; they do not establish future predictive power."
    )


# ==================================================================
# TAB 5 — Validation & evaluator response
# ==================================================================
with tab5:
    st.subheader("Model Validation & Dataset Diagnostics")
    st.caption(
        "This section directly addresses class imbalance, validation methodology, target construction, and temporal-signal limitations."
    )

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Total records", f"{len(df):,}")
    vc2.metric("Low class", f"{(df['Efficiency_Status'] == 'Low').mean() * 100:.2f}%")
    vc3.metric("Medium class", f"{(df['Efficiency_Status'] == 'Medium').mean() * 100:.2f}%")
    vc4.metric("High class", f"{(df['Efficiency_Status'] == 'High').mean() * 100:.2f}%")

    st.markdown("### 1. Baseline and Holdout Comparison")
    if not final_comparison.empty:
        preferred = [
            "method",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
        available = [column for column in preferred if column in final_comparison.columns]
        st.dataframe(
            final_comparison[available].round(4),
            width="stretch",
            hide_index=True,
        )

        plot_frame = final_comparison[["method", "balanced_accuracy", "macro_f1"]].copy()
        plot_frame = plot_frame.melt(
            id_vars="method", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            plot_frame,
            x="method",
            y="Score",
            color="Metric",
            barmode="group",
            title="Balanced Accuracy and Macro F1",
        )
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `03_modeling.py` to generate the final comparison artifact.")

    st.markdown("### 2. Target-Construction Diagnostic")
    if target_diagnostics:
        majority = target_diagnostics.get("majority_baseline", {})
        rule = target_diagnostics.get("business_rule_holdout", {})
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Majority Macro F1", f"{majority.get('macro_f1', float('nan')):.4f}")
        tc2.metric("Rule Holdout Macro F1", f"{rule.get('macro_f1', float('nan')):.4f}")
        tc3.metric(
            "Rule disagreements (100k rows)",
            f"{target_diagnostics.get('business_rule_disagreements', 0):,}",
        )
        st.warning(
            "The transparent rule reproduces the holdout perfectly and disagrees with only two records in the full dataset. "
            "Therefore, near-perfect ML performance primarily reflects target-rule reconstruction."
        )
    else:
        st.info("Target diagnostic artifact is not available.")

    st.markdown("### 3. Expanding-Window Temporal Validation")
    if not temporal_summary.empty:
        st.dataframe(temporal_summary.round(4), width="stretch", hide_index=True)
        if {"model", "macro_f1_mean"}.issubset(temporal_summary.columns):
            fig = px.bar(
                temporal_summary,
                x="model",
                y="macro_f1_mean",
                error_y=("macro_f1_std" if "macro_f1_std" in temporal_summary.columns else None),
                title="Expanding-Window Macro F1",
            )
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `temporal_validation.py` to generate expanding-window results.")

    st.markdown("### 4. Feature Ablation")
    if not ablation_results.empty:
        preferred = [
            "experiment",
            "feature_count",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "high_recall",
        ]
        available = [column for column in preferred if column in ablation_results.columns]
        st.dataframe(
            ablation_results[available].round(4),
            width="stretch",
            hide_index=True,
        )
        if {"experiment", "macro_f1"}.issubset(ablation_results.columns):
            fig = px.bar(
                ablation_results,
                x="experiment",
                y="macro_f1",
                title="Macro F1 After Removing Rule-Linked Features",
            )
            fig.update_yaxes(range=[0, 1.05])
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(fig, width="stretch")
        st.error(
            "Removing all rule-linked features reduces Macro F1 to roughly chance-level performance, demonstrating that sensor/network features do not independently explain the current label in this dataset."
        )
    else:
        st.info("Feature-ablation results are not available.")

    st.markdown("### 5. Temporal-Signal Audit")
    if temporal_diagnostics:
        horizon_rows = temporal_diagnostics.get("horizons", [])
        horizon_frame = pd.DataFrame(horizon_rows)
        if not horizon_frame.empty:
            columns = [
                column
                for column in [
                    "horizon",
                    "observed_same_status_rate",
                    "expected_same_status_if_independent",
                    "agreement_above_chance",
                    "cohen_kappa",
                    "mutual_information",
                ]
                if column in horizon_frame.columns
            ]
            st.dataframe(
                horizon_frame[columns].round(6),
                width="stretch",
                hide_index=True,
            )
        st.warning(
            "Cohen's kappa is approximately zero and observed status agreement is essentially equal to independent chance agreement. "
            "The dataset therefore does not support a defensible future-risk prediction claim."
        )
    else:
        st.info("Temporal-signal diagnostic artifact is not available.")

    st.markdown("### Final Interpretation")
    st.success(
        "The project is positioned as a validated current-state manufacturing efficiency classification and analytics system: "
        "descriptive monitoring + transparent business rule + ML benchmarking + explainability. "
        "Future degradation forecasting is intentionally excluded because the available data does not contain meaningful temporal dependence."
    )


st.markdown("---")
st.caption(
    "Thales Smart Manufacturing analytics prototype · Current-state classification, validation, and operational monitoring · For demonstration and evaluation purposes."
)
