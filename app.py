"""Interactive Streamlit Dashboard for Indian Electricity Demand Forecasting.

Zero-shot foundation models (Google TimesFM 2.5 200M) vs. tuned local model (LightGBM)
and seasonal-naive baselines on all-India electricity demand.
"""
from __future__ import annotations

import pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import main

st.set_page_config(page_title="Indian Electricity Demand Forecasting", layout="wide")

FORECASTS_PATH = main.FORECAST_PATH
SUMMARY_PATH = main.RESULTS_DIR / "summary_detailed.csv"
HORIZON_PATH = main.RESULTS_DIR / "error_by_horizon.csv"
CALIBRATION_PATH = main.RESULTS_DIR / "calibration.csv"
PROBABILISTIC_PATH = main.RESULTS_DIR / "probabilistic.csv"
SIGNIFICANCE_PATH = main.RESULTS_DIR / "significance_dm.csv"
PEAK_PATH = main.RESULTS_DIR / "peak_metrics.csv"


@st.cache_data
def load_series() -> pd.Series | None:
    if main.PROCESSED_SERIES_PATH.exists():
        return main.load_processed()
    return None


@st.cache_data
def load_forecasts() -> pd.DataFrame | None:
    if FORECASTS_PATH.exists():
        return pd.read_parquet(FORECASTS_PATH)
    return None


@st.cache_data
def load_table(path: pathlib.Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


st.title("Zero-Shot Foundation Models vs. Tuned Local Model")
st.caption(
    f"24-hour-ahead all-India electricity demand forecasting. "
    f"Dataset: Mendeley DOI [{main.MENDELEY_DOI}](https://doi.org/{main.MENDELEY_DOI}) (CC BY 4.0), via Grid-India SCADA."
)

# --------------------------------------------------------------------------------------
# 1. Headline Benchmark Results
# --------------------------------------------------------------------------------------
st.header("Point Forecast Accuracy")
summary = load_table(SUMMARY_PATH)
if summary is not None:
    display_cols = [c for c in ["model", "MAE", "RMSE", "sMAPE", "MAPE", "MASE", "MAE_skill_%", "RMSE_skill_%"] if c in summary.columns]
    st.dataframe(summary[display_cols].round(3), use_container_width=True)
    st.caption(
        "MASE is scaled by the in-sample MAE of the stronger seasonal naive on the training split "
        "(4,445.2 MW, m=24), frozen and applied identically across all tiers. MASE < 1 means beating in-sample naive."
    )
else:
    st.info("Results tables will appear here once experiments have run.")

# --------------------------------------------------------------------------------------
# 2. Interactive 24-Hour Forecast Explorer
# --------------------------------------------------------------------------------------
st.header("24-Hour Rolling Forecast Explorer")
fc = load_forecasts()
y = load_series()

if fc is not None and y is not None:
    models = sorted(fc["model"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        default_model = "timesfm_ctx896" if "timesfm_ctx896" in models else ("timesfm" if "timesfm" in models else models[0])
        selected_model = st.selectbox("Model Tier", models, index=models.index(default_model))
    sub = fc[fc["model"] == selected_model]
    origins = sorted(sub["origin"].unique())
    with c2:
        origin = st.select_slider("Forecast Origin (every 6 hours)", options=origins, value=origins[len(origins) // 2])

    window = sub[sub["origin"] == origin].sort_values("h")
    context = y.loc[:origin].iloc[-168:]

    fig = go.Figure()
    # 7-day history
    fig.add_trace(go.Scatter(x=context.index, y=context.to_numpy(), name="History (Past 7 Days)", line=dict(color="#888", width=1.5)))

    # Quantile fan chart
    for lo_q, hi_q, level in reversed(main.INTERVAL_LEVELS):
        lo_col, hi_col = f"q{lo_q:g}", f"q{hi_q:g}"
        if lo_col in window.columns and hi_col in window.columns and not window[lo_col].isna().all():
            shade = 0.10 + 0.16 * (1.0 - level)
            fig.add_trace(go.Scatter(x=window["target_time"], y=window[hi_col], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=window["target_time"], y=window[lo_col],
                name=f"P{int(lo_q*100)}–P{int(hi_q*100)}", fill="tonexty",
                fillcolor=f"rgba(99,110,250,{shade:.2f})", line=dict(width=0), hoverinfo="skip"
            ))

    # Actual vs Prediction
    fig.add_trace(go.Scatter(x=window["target_time"], y=window["actual"], name="Actual Demand", line=dict(color="#111", width=2.5)))
    fig.add_trace(go.Scatter(x=window["target_time"], y=window["prediction"], name="Forecast", line=dict(color="#EF553B", width=2.5, dash="dash")))
    fig.update_layout(height=460, yaxis_title="Demand (MW)", xaxis_title="Target Timestamp (IST)", hovermode="x unified", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    peak_i = window["prediction"].idxmax()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Peak", f"{window.loc[peak_i, 'prediction']:,.0f} MW")
    m2.metric("Predicted Peak Hour", pd.Timestamp(window.loc[peak_i, "target_time"]).strftime("%H:%M IST"))
    m3.metric("Actual Peak", f"{window['actual'].max():,.0f} MW")
    w_mae = (window["actual"] - window["prediction"]).abs().mean()
    m4.metric("Window MAE", f"{w_mae:,.0f} MW")
else:
    st.info("Interactive rolling forecast viewer will activate once `forecasts.parquet` is generated via `python main.py --stage all`.")

# --------------------------------------------------------------------------------------
# 3. Error by Forecast Horizon
# --------------------------------------------------------------------------------------
st.header("Error by Forecast Horizon (h = 1 to 24)")
curve = load_table(HORIZON_PATH)
if curve is not None:
    fig_h = go.Figure()
    for name, d in curve.groupby("model"):
        fig_h.add_trace(go.Scatter(x=d["h"], y=d["MAE"], name=name, mode="lines+markers"))
    fig_h.update_layout(height=380, xaxis_title="Hours Ahead (h)", yaxis_title="MAE (MW)", hovermode="x unified", margin=dict(t=30))
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("TimesFM dominates at short-to-medium horizons. Seasonal naives remain flat across the horizon.")

# --------------------------------------------------------------------------------------
# 4. Daily Peak Demand Analysis
# --------------------------------------------------------------------------------------
st.header("Daily Peak Demand Accuracy")
peak_df = load_table(PEAK_PATH)
if peak_df is not None:
    st.dataframe(peak_df.round(2), use_container_width=True)
    st.caption(
        "Peak bias is signed as Actual - Forecast: positive means systematic under-prediction. "
        "LightGBM under-predicts peak demand by ~4,918 MW due to mean regression, while TimesFM captures the peak within 631 MW."
    )

# --------------------------------------------------------------------------------------
# 5. Interval Calibration & Distributional Scores
# --------------------------------------------------------------------------------------
cal_df = load_table(CALIBRATION_PATH)
prob_df = load_table(PROBABILISTIC_PATH)

if cal_df is not None or prob_df is not None:
    st.header("Probabilistic Forecasting & Interval Calibration")
    c_cal, c_prob = st.columns([1, 1])

    with c_cal:
        if cal_df is not None and not cal_df.empty:
            st.subheader("Decile Reliability Diagram")
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Ideal Calibration", line=dict(color="#999", dash="dash")))
            for name, d in cal_df.groupby("model"):
                d = d.sort_values("nominal")
                fig_cal.add_trace(go.Scatter(x=d["nominal"], y=d["empirical"], name=name, mode="lines+markers"))
            fig_cal.update_layout(height=380, xaxis_title="Nominal Quantile", yaxis_title="Empirical Exceedance Fraction", hovermode="x unified", margin=dict(t=30))
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption("Points on the diagonal indicate perfect calibration. Curves below the diagonal indicate overconfidence.")

    with c_prob:
        if prob_df is not None and not prob_df.empty:
            st.subheader("Distributional Scores (CRPS & Winkler)")
            st.dataframe(prob_df.round(3), use_container_width=True)
            st.caption("CRPS integrates over the 9 deciles. Winkler score rewards sharpness while heavily penalizing misses.")

# --------------------------------------------------------------------------------------
# 6. Statistical Significance
# --------------------------------------------------------------------------------------
dm_df = load_table(SIGNIFICANCE_PATH)
if dm_df is not None and not dm_df.empty:
    st.header("Statistical Significance (Diebold-Mariano HAC)")
    cols = [c for c in ["model_A", "model_B", "winner", "dm_stat", "p_value", "significant_1pct", "significant_5pct"] if c in dm_df.columns]
    st.dataframe(dm_df[cols].round(4), use_container_width=True)
    st.caption(
        "Pairwise Diebold-Mariano tests on origin-level absolute errors using Newey-West HAC variance estimator "
        "and Harvey small-sample correction. Accounting for window overlap autocorrelation is essential for valid inference."
    )

# --------------------------------------------------------------------------------------
# 7. Historical Series Explorer
# --------------------------------------------------------------------------------------
if y is not None:
    st.header("Historical All-India Demand (46 Months)")
    fig_hist = go.Figure()
    daily_s = y.resample("D").mean()
    fig_hist.add_trace(go.Scatter(x=daily_s.index, y=daily_s.to_numpy(), name="Daily Mean Demand", line=dict(color="#636EFA", width=1.2)))
    fig_hist.update_layout(height=320, yaxis_title="Demand (MW)", margin=dict(t=30))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"{y.index.min():%Y-%m-%d} to {y.index.max():%Y-%m-%d} — {len(y):,} hourly observations (IST).")
