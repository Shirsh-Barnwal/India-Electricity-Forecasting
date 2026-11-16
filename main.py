"""Indian Electricity Demand Forecasting Pipeline.

A compact, unified, reproducible implementation comparing zero-shot foundation models
(Google TimesFM 2.5 200M) against a tuned local model (LightGBM direct multi-horizon)
and seasonal-naive baselines on all-India electricity demand.

Dataset: Mendeley Data (DOI 10.17632/y58jknpgs8.2), CC BY 4.0, via Grid-India SCADA.
Evaluation: 24h horizon, 6h rolling stride, 832 identical test origins, chronological 70/15/15 splits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import json
import math
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

import holidays
import lightgbm as lgb
import numpy as np
from openpyxl import load_workbook
import pandas as pd
from scipy import stats

# ======================================================================================
# 1. CONFIGURATION & CONSTANTS
# ======================================================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MENDELEY_RAW_DIR = RAW_DIR / "mendeley"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "reference_results"
EDA_DIR = RESULTS_DIR / "eda"

MANIFEST_PATH = RAW_DIR / "mendeley_manifest.json"
PROCESSED_SERIES_PATH = PROCESSED_DIR / "demand_hourly.parquet"
GAP_REPORT_PATH = PROCESSED_DIR / "preprocessing_report.json"
FORECAST_PATH = RESULTS_DIR / "forecasts.parquet"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
HORIZON_PATH = RESULTS_DIR / "error_by_horizon.csv"
META_PATH = RESULTS_DIR / "run_metadata.json"
TUNING_PATH = RESULTS_DIR / "lightgbm_tuning.json"

for _d in (RAW_DIR, MENDELEY_RAW_DIR, PROCESSED_DIR, RESULTS_DIR, EDA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Dataset details
MENDELEY_DATASET_ID = "y58jknpgs8"
MENDELEY_VERSION = 2
MENDELEY_DOI = "10.17632/y58jknpgs8.2"
MENDELEY_MANIFEST_URL = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}/files"
    f"?folder_id=root&version={MENDELEY_VERSION}"
)

TIMEZONE = "Asia/Kolkata"  # IST, UTC+05:30. India observes no DST.
SCADA_DEMAND_COL = 10      # Column index of NLDC_DEMAND|P in sheet Sheet1
SCADA_TIME_COL = 0
SCADA_HEADER_ROWS = 2
SCADA_SHEET = "Sheet1"
HOURLY_SHEET = "Report"
HOURLY_FILE_NAME = "January 2024- June 2025.xlsx"

PLAUSIBLE_MIN_MW = 20_000.0
PLAUSIBLE_MAX_MW = 300_000.0

# Forecasting parameters
HORIZON = 24
SEASONAL_PERIOD_DAILY = 24
SEASONAL_PERIOD_WEEKLY = 168
ORIGIN_STRIDE_HOURS = 6

CONTEXT_LENGTHS = (168, 336, 720, 896, 1024)
DEFAULT_CONTEXT = 896
MIN_CONTEXT = 896  # Longest context evaluated in test comparison

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_COLS = [f"q{q:g}" for q in QUANTILES]
INTERVAL_LEVELS = (
    (0.4, 0.6, 0.2),
    (0.3, 0.7, 0.4),
    (0.2, 0.8, 0.6),
    (0.1, 0.9, 0.8),
)
NOMINAL_COVERAGE = 0.8

RANDOM_SEED = 42

TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MAX_CONTEXT = 1024
TIMESFM_MAX_HORIZON = 256

MODEL_LABELS = {
    "seasonal_naive_daily": "Seasonal naive (m=24)",
    "seasonal_naive_weekly": "Seasonal naive (m=168)",
    "lightgbm": "LightGBM",
    "timesfm": "TimesFM 2.5 (zero-shot)",
    "timesfm_ctx896": "TimesFM 2.5, ctx 896 (zero-shot)",
}

# ======================================================================================
# 2. DATA INGESTION & PREPROCESSING
# ======================================================================================
def _curl(args: list[str]) -> bytes:
    """Execute curl subprocess. Cloudflare rejects python-requests TLS fingerprint."""
    proc = subprocess.run(["curl", "-sS", "--fail", *args], capture_output=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:400]}"
        )
    return proc.stdout


def fetch_manifest(force: bool = False) -> list[dict[str, Any]]:
    """Fetch dataset file manifest from Mendeley Data API."""
    if MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw = _curl(["--max-time", "120", MENDELEY_MANIFEST_URL])
    manifest = json.loads(raw)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_data(verbose: bool = True) -> list[str]:
    """Download all 29 Mendeley workbooks idempotently. Verifies file sizes."""
    manifest = fetch_manifest()
    names: list[str] = []
    for entry in sorted(manifest, key=lambda e: e["filename"]):
        name = entry["filename"]
        dest = MENDELEY_RAW_DIR / name
        expected = entry["size"]
        names.append(name)
        if dest.exists() and dest.stat().st_size == expected:
            if verbose:
                print(f"  skip {name} (already complete)", flush=True)
            continue
        if verbose:
            print(f"  downloading {name} ({expected / 1e6:.1f} MB) ...", flush=True)
        _curl(["-L", "--max-time", "3600", "-o", str(dest), entry["content_details"]["download_url"]])
        got = dest.stat().st_size
        if got != expected:
            raise RuntimeError(f"{name}: downloaded {got} bytes, manifest specifies {expected}")
    return names


def _sampling_interval_seconds(path: pathlib.Path) -> float | None:
    """Read native sample interval from hidden '_osi_config' sheet."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if "_osi_config" not in wb.sheetnames:
            return None
        ws = wb["_osi_config"]
        for row in ws.iter_rows(max_row=2, max_col=1, values_only=True):
            cell = row[0]
            if isinstance(cell, str) and '"r"' in cell:
                m = re.search(r'"r"\s*:\s*([0-9.]+)', cell)
                if m:
                    return float(m.group(1))
        return None
    finally:
        wb.close()


def _read_scada_workbook(path: pathlib.Path) -> pd.Series:
    """Read timestamp and NLDC_DEMAND|P from Regime A/B SCADA workbook."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[SCADA_SHEET]
        stamps: list = []
        values: list = []
        for row in ws.iter_rows(min_row=SCADA_HEADER_ROWS + 1, values_only=True):
            ts = row[SCADA_TIME_COL]
            if ts is None:
                continue
            stamps.append(ts)
            values.append(row[SCADA_DEMAND_COL])
    finally:
        wb.close()
    return pd.Series(
        pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
        index=pd.to_datetime(pd.Series(stamps)),
        name="demand",
    )


def _read_hourly_workbook(path: pathlib.Path) -> pd.Series:
    """Read Regime C pre-aggregated hourly workbook.
    
    Drops 6 trailing Excel summary rows (Min/Max/Avg/Sum) and enforces dayfirst=True
    for parsing DD-MM-YYYY timestamps correctly.
    """
    df = pd.read_excel(path, sheet_name=HOURLY_SHEET)
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    demand = pd.to_numeric(df["Demand (MW)"], errors="coerce")
    keep = ts.notna()
    dropped = int((~keep).sum())
    if dropped:
        print(f"      dropped {dropped} trailing non-timestamp rows (Excel summary block)")
    return pd.Series(demand[keep].to_numpy(), index=pd.DatetimeIndex(ts[keep]), name="demand")


def mask_invalid(s: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Mask physically implausible readings and SCADA 0.0 MW telemetry dropouts to NaN."""
    zeros = int((s == 0).sum())
    out = s.replace(0.0, np.nan)
    implausible = int(((out < PLAUSIBLE_MIN_MW) | (out > PLAUSIBLE_MAX_MW)).sum())
    out = out.where((out >= PLAUSIBLE_MIN_MW) & (out <= PLAUSIBLE_MAX_MW))
    return out, {"zero_dropouts": zeros, "implausible": implausible}


def to_hourly(s: pd.Series) -> pd.Series:
    """Aggregate sub-hourly series to hourly mean, matching depositor's Regime C convention."""
    return s.resample("h").mean()


def preprocess_data(verbose: bool = True) -> pd.Series:
    """Process raw workbooks into a clean hourly demand series saved to Parquet."""
    files = sorted(MENDELEY_RAW_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel workbooks found in {MENDELEY_RAW_DIR}. Run with --download first.")

    pieces: list[pd.Series] = []
    stats_list: list[dict] = []
    for path in files:
        t0 = time.time()
        interval = _sampling_interval_seconds(path)
        is_hourly_file = path.name == HOURLY_FILE_NAME
        if is_hourly_file:
            raw = _read_hourly_workbook(path)
            regime = "C (pre-aggregated hourly)"
        else:
            raw = _read_scada_workbook(path)
            regime = f"{'A' if interval == 300.0 else 'B'} (SCADA {interval:g}s)"

        n_raw = len(raw)
        dupes = int(raw.index.duplicated().sum())
        if dupes:
            raw = raw[~raw.index.duplicated(keep="first")]
        raw = raw.sort_index()

        cleaned, flags = mask_invalid(raw)
        hourly = cleaned if is_hourly_file else to_hourly(cleaned)

        pieces.append(hourly)
        stats_list.append({
            "file": path.name,
            "regime": regime,
            "native_interval_s": interval,
            "raw_rows": n_raw,
            "duplicate_timestamps": dupes,
            "hourly_rows": int(len(hourly)),
            "nan_hours": int(hourly.isna().sum()),
            **flags,
            "start": str(hourly.index.min()),
            "end": str(hourly.index.max()),
        })
        if verbose:
            print(f"  {path.name:<32} {regime:<22} {n_raw:>7,} -> {len(hourly):>5,} h ({time.time()-t0:.1f}s)", flush=True)

    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep="first")]

    # Reindex onto a strict hourly grid so gaps become explicit NaNs
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    before = len(series)
    series = series.reindex(full_index)
    missing_idx = series.index[series.isna()]

    print(f"\nHourly Grid: {full_index[0]} -> {full_index[-1]} ({len(full_index):,} hours)")
    print(f"  Observed hours: {before:,}")
    print(f"  Missing hours: {len(missing_idx):,} ({len(missing_idx)/len(full_index)*100:.3f}%)")

    # Past-only forward fill
    filled = series.ffill()
    n_leading = int(filled.isna().sum())
    if n_leading:
        filled = filled.iloc[n_leading:]
        print(f"  Dropped {n_leading} leading hours with no prior observation")

    filled = filled.tz_localize(TIMEZONE)
    out = filled.rename("demand").to_frame()
    out.index.name = "timestamp"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_SERIES_PATH)

    report = {
        "source_doi": MENDELEY_DOI,
        "timezone": TIMEZONE,
        "aggregation": "hourly mean (matches depositor rule for regime C)",
        "gap_fill": "forward fill (past information only)",
        "grid_start": str(full_index[0]),
        "grid_end": str(full_index[-1]),
        "grid_hours": int(len(full_index)),
        "observed_hours": int(before),
        "missing_hours_filled": int(len(missing_idx)),
        "missing_hours_list": [str(t) for t in missing_idx[:200]],
        "final_rows": int(len(filled)),
        "per_file": stats_list,
    }
    GAP_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved processed series to {PROCESSED_SERIES_PATH} ({len(filled):,} rows)")
    print(f"Peak demand: {filled.max():,.1f} MW at {filled.idxmax()}")
    return filled["demand"]


def load_processed() -> pd.Series:
    """Load the processed hourly demand series."""
    if not PROCESSED_SERIES_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_SERIES_PATH} not found. Run with --preprocess first.")
    df = pd.read_parquet(PROCESSED_SERIES_PATH)
    return df["demand"]

# ======================================================================================
# 3. SPLITS & ROLLING-ORIGIN HARNESS
# ======================================================================================
@dataclass(frozen=True)
class Splits:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train[-1]

    @property
    def val_end(self) -> pd.Timestamp:
        return self.val[-1]

    def describe(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            idx = getattr(self, name)
            rows.append({
                "split": name,
                "start": idx[0],
                "end": idx[-1],
                "hours": len(idx),
                "days": round(len(idx) / 24, 1),
                "share_%": round(100 * len(idx) / (len(self.train) + len(self.val) + len(self.test)), 1),
            })
        return pd.DataFrame(rows)


def chronological_splits(y: pd.Series, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> Splits:
    """Chronological 70/15/15 train/val/test split. Never random."""
    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    idx = y.index
    splits = Splits(
        train=idx[:n_train],
        val=idx[n_train:n_train + n_val],
        test=idx[n_train + n_val:],
    )
    assert splits.train[-1] < splits.val[0] < splits.val[-1] < splits.test[0], "Splits must be strictly ordered in time"
    return splits


def make_origins(y: pd.Series, start=None, end=None, stride_hours: int = ORIGIN_STRIDE_HOURS,
                 min_context: int = MIN_CONTEXT, horizon: int = HORIZON) -> pd.DatetimeIndex:
    """Generate rolling forecast origins within [start, end] having sufficient history and horizon."""
    idx = y.index
    lo = idx[min_context - 1]
    hi = idx[-horizon - 1]
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    if end is not None:
        hi = min(hi, pd.Timestamp(end))
    candidates = idx[(idx >= lo) & (idx <= hi)]
    return pd.DatetimeIndex(candidates[::stride_hours])

# ======================================================================================
# 4. FEATURE ENGINEERING
# ======================================================================================
LAGS: tuple[int, ...] = (0, 1, 2, 3, 24, 48, 168, 336)
ROLL_WINDOWS: tuple[int, ...] = (24, 168)
MAX_LAG = max(max(LAGS), max(ROLL_WINDOWS))

ORIGIN_FEATURES: tuple[str, ...] = tuple(
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["diff_24", "diff_168"]
)

CALENDAR_FEATURES: tuple[str, ...] = (
    "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
)

FEATURE_NAMES: tuple[str, ...] = ORIGIN_FEATURES + CALENDAR_FEATURES


@functools.lru_cache(maxsize=1)
def _india_holidays(first_year: int, last_year: int) -> set:
    cal = holidays.India(years=range(first_year, last_year + 1))
    return set(cal.keys())


def origin_features(y: pd.Series) -> pd.DataFrame:
    """Construct features known at origin T using y(<= T) only. Strictly trailing."""
    out = pd.DataFrame(index=y.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    for w in ROLL_WINDOWS:
        roll = y.rolling(window=w, min_periods=w)
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
    out["diff_24"] = y - y.shift(24)
    out["diff_168"] = y - y.shift(168)
    return out


def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar attributes of target timestamps T+h (known in advance)."""
    idx = pd.DatetimeIndex(times)
    hol = _india_holidays(int(idx.year.min()), int(idx.year.max()))
    date_objs = pd.Index([d.date() for d in idx.normalize()])

    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = date_objs.isin(hol).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def training_frame(y: pd.Series, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble (X, target) for a direct horizon h, indexed by origin T."""
    base = origin_features(y)
    target_times = y.index + pd.Timedelta(hours=horizon)
    cal = calendar_features(target_times)
    cal.index = y.index

    X = pd.concat([base, cal], axis=1)[list(FEATURE_NAMES)]
    target = pd.Series(y.shift(-horizon).to_numpy(), index=y.index, name="target")

    ok = X.notna().all(axis=1) & target.notna()
    return X[ok], target[ok]


def features_at_origin(history: pd.Series, horizon: int) -> pd.DataFrame:
    """Build single feature row at the end of history for step horizon."""
    if len(history) < MAX_LAG + 1:
        raise ValueError(f"Need at least {MAX_LAG + 1} history points, got {len(history)}")
    base = origin_features(history).iloc[[-1]]
    origin = history.index[-1]
    cal = calendar_features(pd.DatetimeIndex([origin + pd.Timedelta(hours=horizon)]))
    cal.index = base.index
    return pd.concat([base, cal], axis=1)[list(FEATURE_NAMES)]

# ======================================================================================
# 5. FORECASTING MODELS
# ======================================================================================
# --- 5.1 Seasonal Naive Baselines ---
class SeasonalNaive:
    def __init__(self, seasonal_period: int, name: str | None = None) -> None:
        self.m = int(seasonal_period)
        self.name = name or f"seasonal_naive_m{self.m}"

    @property
    def min_context(self) -> int:
        return self.m

    def forecast(self, history: pd.Series | np.ndarray, horizon: int = HORIZON) -> np.ndarray:
        y = np.asarray(history, dtype=float).ravel()
        if y.size < self.m:
            raise ValueError(f"{self.name}: need at least {self.m} observations, got {y.size}")
        if horizon > self.m:
            raise ValueError(f"{self.name}: horizon {horizon} exceeds period {self.m}")
        return y[-self.m:][:horizon].copy()

    def fit(self, train: pd.Series) -> SeasonalNaive:
        return self


def daily_naive() -> SeasonalNaive:
    return SeasonalNaive(SEASONAL_PERIOD_DAILY, name="seasonal_naive_daily")


def weekly_naive() -> SeasonalNaive:
    return SeasonalNaive(SEASONAL_PERIOD_WEEKLY, name="seasonal_naive_weekly")


def check_weekly_seasonality(train_vals: pd.Series) -> dict:
    """Verify lag-168 is a local minimum vs lags 144/192 to confirm weekly structure exists."""
    v = train_vals.to_numpy(dtype=float)
    lag_mae = {m: float(np.nanmean(np.abs(v[m:] - v[:-m]))) for m in (24, 120, 144, 168, 192)}
    is_local_min = lag_mae[168] < lag_mae[144] and lag_mae[168] < lag_mae[192]

    dow = train_vals.groupby(train_vals.index.dayofweek).mean()
    dow_spread_pct = float((dow.max() - dow.min()) / dow.mean() * 100)
    hod = train_vals.groupby(train_vals.index.hour).mean()
    hod_spread_pct = float((hod.max() - hod.min()) / hod.mean() * 100)

    return {
        "lag_mae": lag_mae,
        "lag168_is_local_minimum": is_local_min,
        "dow_spread_pct": dow_spread_pct,
        "hod_spread_pct": hod_spread_pct,
    }


# --- 5.2 LightGBM Direct Multi-Horizon Forecaster ---
LGB_BASE_PARAMS: dict = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "n_jobs": 6,
    "verbose": -1,
    "seed": RANDOM_SEED,
}

LGB_PARAM_GRID: tuple[dict, ...] = (
    {"num_leaves": 7, "learning_rate": 0.05},
    {"num_leaves": 15, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.05},
    {"num_leaves": 63, "learning_rate": 0.05},
    {"num_leaves": 127, "learning_rate": 0.05},
    {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 800},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 800},
    {"num_leaves": 15, "learning_rate": 0.10},
)

TUNING_HORIZONS: tuple[int, ...] = (1, 6, 12, 18, 24)


class LightGBMForecaster:
    """Direct multi-horizon LightGBM with 24 horizon regressors and quantile regressors."""

    def __init__(self, horizon: int = HORIZON, params: dict | None = None, quantile: float | None = None) -> None:
        self.horizon = int(horizon)
        self.params = {**LGB_BASE_PARAMS, **(params or {})}
        self.quantile = quantile
        if quantile is not None:
            self.params = {**self.params, "objective": "quantile", "alpha": float(quantile), "metric": "quantile"}
        self.name = "lightgbm" if quantile is None else f"lightgbm_q{quantile:g}"
        self.models_: dict[int, lgb.LGBMRegressor] = {}

    @property
    def min_context(self) -> int:
        return MAX_LAG + 1

    def fit(self, y_full: pd.Series, train_index: pd.DatetimeIndex, verbose: bool = False) -> LightGBMForecaster:
        for h in range(1, self.horizon + 1):
            X, target = training_frame(y_full, h)
            mask = X.index.isin(train_index)
            model = lgb.LGBMRegressor(**self.params)
            model.fit(X[mask], target[mask])
            self.models_[h] = model
            if verbose:
                print(f"    h={h:>2} fitted on {int(mask.sum()):,} origins", flush=True)
        return self

    def forecast(self, history: pd.Series, horizon: int | None = None) -> np.ndarray:
        h_max = int(horizon or self.horizon)
        if not self.models_:
            raise RuntimeError("Model is not fitted")
        preds = np.empty(h_max, dtype=float)
        for h in range(1, h_max + 1):
            row = features_at_origin(history, h)
            preds[h - 1] = float(self.models_[h].predict(row)[0])
        return preds

    def forecast_batch(self, y_full: pd.Series, origins: Iterable[pd.Timestamp], horizon: int | None = None) -> np.ndarray:
        h_max = int(horizon or self.horizon)
        origins_idx = pd.DatetimeIndex(origins)
        out = np.full((len(origins_idx), h_max), np.nan, dtype=float)
        for h in range(1, h_max + 1):
            base = origin_features(y_full)
            cal = calendar_features(y_full.index + pd.Timedelta(hours=h))
            cal.index = y_full.index
            Xf = pd.concat([base, cal], axis=1)[list(FEATURE_NAMES)]
            rows = Xf.reindex(origins_idx)
            valid = rows.notna().all(axis=1).to_numpy()
            if valid.any():
                out[valid, h - 1] = self.models_[h].predict(rows[valid])
        return out


def tune_lightgbm(y_full: pd.Series, train_index: pd.DatetimeIndex, val_index: pd.DatetimeIndex,
                  verbose: bool = True) -> tuple[dict, list[dict]]:
    """Grid search over LightGBM configurations on validation split."""
    records: list[dict] = []
    for i, override in enumerate(LGB_PARAM_GRID, 1):
        t0 = time.time()
        maes: list[float] = []
        for h in TUNING_HORIZONS:
            X, target = training_frame(y_full, h)
            tr = X.index.isin(train_index)
            va = X.index.isin(val_index)
            model = lgb.LGBMRegressor(**{**LGB_BASE_PARAMS, **override})
            model.fit(X[tr], target[tr])
            pred = model.predict(X[va])
            maes.append(float(np.mean(np.abs(target[va].to_numpy() - pred))))
        mean_mae = float(np.mean(maes))
        records.append({"params": override, "val_mae": mean_mae, "per_horizon_mae": dict(zip(TUNING_HORIZONS, maes))})
        if verbose:
            print(f"  [{i}/{len(LGB_PARAM_GRID)}] {override} -> val MAE {mean_mae:,.1f} MW ({time.time()-t0:.1f}s)", flush=True)
    best = min(records, key=lambda r: r["val_mae"])
    if verbose:
        print(f"  Best params: {best['params']} (val MAE: {best['val_mae']:,.1f} MW)")
    return best["params"], records


# --- 5.3 Google TimesFM 2.5 (200M) Zero-Shot Forecaster ---
_QUANTILE_COLUMN = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9}
_TIMESFM_CACHE: dict[str, object] = {}


class TimesFMForecaster:
    """Zero-shot TimesFM 2.5 (200M) univariate CPU forecaster."""

    def __init__(self, context_length: int = DEFAULT_CONTEXT, checkpoint: str = TIMESFM_CHECKPOINT,
                 horizon: int = HORIZON, batch_size: int = 32) -> None:
        self.context_length = int(context_length)
        self.checkpoint = checkpoint
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.name = f"timesfm_ctx{self.context_length}"
        self._model = None

    @property
    def min_context(self) -> int:
        return self.context_length

    def load(self):
        """Compile and cache checkpoint."""
        if self._model is not None:
            return self._model
        if self.checkpoint in _TIMESFM_CACHE:
            self._model = _TIMESFM_CACHE[self.checkpoint]
            return self._model
        import timesfm

        t0 = time.time()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=TIMESFM_MAX_CONTEXT,
                max_horizon=TIMESFM_MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        print(f"  Loaded {self.checkpoint} in {time.time()-t0:.1f}s")
        self._verify_quantile_layout()
        _TIMESFM_CACHE[self.checkpoint] = model
        return model

    def _verify_quantile_layout(self) -> None:
        """Assert quantile output layout: [mean, q10..q90]."""
        probe = np.linspace(100.0, 200.0, 512) + np.sin(np.arange(512) / 3.0) * 5.0
        point, quant = self._model.forecast(horizon=8, inputs=[probe])
        quant = np.asarray(quant)
        if quant.ndim != 3 or quant.shape[2] != 10:
            raise RuntimeError(f"Unexpected quantile shape {quant.shape}; expected (b, h, 10)")
        q10, q50, q90 = quant[0, :, 1], quant[0, :, 5], quant[0, :, 9]
        if not (np.all(q10 <= q50 + 1e-6) and np.all(q50 <= q90 + 1e-6)):
            raise RuntimeError("Quantile columns not ordered q10 <= q50 <= q90")
        if not np.allclose(np.asarray(point)[0], quant[0, :, 0], rtol=0.05, atol=1.0):
            raise RuntimeError("Column 0 of quantile output does not match point forecast")

    def fit(self, *args, **kwargs) -> TimesFMForecaster:
        return self  # Zero-shot, strictly no training

    def forecast(self, history: pd.Series | np.ndarray, horizon: int | None = None) -> np.ndarray:
        point, _ = self.forecast_with_quantiles([history], horizon)
        return point[0]

    def forecast_with_quantiles(self, histories: Iterable, horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        model = self.load()
        h = int(horizon or self.horizon)
        inputs = [np.asarray(x, dtype=np.float32).ravel()[-self.context_length:] for x in histories]

        points: list[np.ndarray] = []
        quants: list[np.ndarray] = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i: i + self.batch_size]
            p, q = model.forecast(horizon=h, inputs=chunk)
            points.append(np.asarray(p, dtype=float))
            quants.append(np.asarray(q, dtype=float))

        point = np.concatenate(points, axis=0)
        quant_full = np.concatenate(quants, axis=0)
        cols = [_QUANTILE_COLUMN[q] for q in QUANTILES]
        return point, quant_full[:, :, cols]

    def forecast_windows(self, y: pd.Series, origins: pd.DatetimeIndex, horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
        values = y.to_numpy(dtype=float)
        positions = y.index.get_indexer(pd.DatetimeIndex(origins))
        if (positions < 0).any():
            raise ValueError("Some origins not present in series index")

        histories = []
        for p in positions:
            start = p + 1 - self.context_length
            if start < 0:
                raise ValueError(f"Origin at {p} has {p+1} history points, needs {self.context_length}")
            histories.append(values[start: p + 1])
        return self.forecast_with_quantiles(histories, horizon)

# ======================================================================================
# 6. EVALUATION METRICS & STATISTICAL TESTS
# ======================================================================================
def _as_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("Empty input")
    return a, p


def mae(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(actual, predicted) -> float:
    """0-200% symmetric MAPE convention."""
    a, p = _as_arrays(actual, predicted)
    denom = np.abs(a) + np.abs(p)
    ratio = np.divide(np.abs(a - p), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(200.0 * ratio))


def mape(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    ok = a != 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100.0)


def seasonal_naive_scale(train_series, seasonal_period: int) -> float:
    """In-sample MAE of seasonal naive on training data (MASE denominator)."""
    y = np.asarray(train_series, dtype=float).ravel()
    m = int(seasonal_period)
    if y.size <= m:
        raise ValueError(f"Need > {m} observations, got {y.size}")
    diffs = np.abs(y[m:] - y[:-m])
    diffs = diffs[~np.isnan(diffs)]
    scale = float(np.mean(diffs))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Seasonal naive scale is non-finite or 0")
    return scale


def mase(actual, predicted, scale: float) -> float:
    return mae(actual, predicted) / float(scale)


def evaluate_forecast(actual, predicted, scale: float | None = None) -> dict[str, float]:
    out = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "MAPE": mape(actual, predicted),
    }
    out["MASE"] = mase(actual, predicted, scale) if scale is not None else float("nan")
    return out


# --- Probabilistic Metrics ---
def pinball_loss(actual, predicted_quantile, tau: float) -> float:
    a, p = _as_arrays(actual, predicted_quantile)
    diff = a - p
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def picp(actual, lower, upper) -> float:
    a, lo = _as_arrays(actual, lower)
    _, hi = _as_arrays(actual, upper)
    return float(np.mean((a >= lo) & (a <= hi)))


def interval_width(lower, upper) -> float:
    lo, hi = _as_arrays(lower, upper)
    return float(np.mean(hi - lo))


def winkler_score(actual, lower, upper, alpha: float = 0.2) -> float:
    a, lo = _as_arrays(actual, lower)
    _, hi = _as_arrays(actual, upper)
    width = hi - lo
    penalty = np.zeros_like(width)
    below = a < lo
    above = a > hi
    penalty[below] = (2.0 / alpha) * (lo[below] - a[below])
    penalty[above] = (2.0 / alpha) * (a[above] - hi[above])
    return float(np.mean(width + penalty))


def crps_from_quantiles(actual, quantile_preds, taus) -> float:
    a = np.asarray(actual, dtype=float).ravel()
    taus_arr = np.asarray(taus, dtype=float).ravel()
    q = np.asarray(quantile_preds, dtype=float)
    if q.ndim != 2 or q.shape[1] != taus_arr.size:
        raise ValueError(f"quantile_preds must be (n, {taus_arr.size}), got {q.shape}")
    losses = [pinball_loss(a, q[:, j], float(t)) for j, t in enumerate(taus_arr)]
    return float(2.0 * np.mean(losses))


def rearrange_quantiles(quantile_preds) -> np.ndarray:
    """Sort predicted quantiles row-wise to enforce monotonicity (Chernozhukov et al. 2010)."""
    q = np.asarray(quantile_preds, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {q.shape}")
    return np.sort(q, axis=1)


def count_quantile_crossings(quantile_preds) -> dict[str, float]:
    q = np.asarray(quantile_preds, dtype=float)
    diffs = np.diff(q, axis=1)
    rows = (diffs < 0).any(axis=1)
    return {
        "rows_with_crossing": int(rows.sum()),
        "fraction_of_rows": float(rows.mean()) if rows.size else 0.0,
        "max_inversion": float(np.abs(np.minimum(diffs, 0)).max()) if diffs.size else 0.0,
    }


def quantile_calibration(actual, quantile_preds, taus) -> dict[float, float]:
    a = np.asarray(actual, dtype=float).ravel()
    q = np.asarray(quantile_preds, dtype=float)
    taus_arr = np.asarray(taus, dtype=float).ravel()
    return {float(t): float(np.mean(a <= q[:, j])) for j, t in enumerate(taus_arr)}


# --- Diebold-Mariano Significance Test (HAC Newey-West) ---
def newey_west_variance(d: np.ndarray, max_lag: int | None = None) -> float:
    """Long-run variance with Bartlett kernel."""
    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    if max_lag is None:
        max_lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), n - 2))

    d_centred = d - d.mean()
    gamma0 = float(np.dot(d_centred, d_centred) / n)
    total = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(d_centred[lag:], d_centred[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        total += 2.0 * weight * cov
    return total if total > 0 else gamma0


def diebold_mariano(actual, pred_a, pred_b, power: int = 1, max_lag: int | None = None,
                    harvey_correction: bool = True) -> dict[str, float]:
    a = np.asarray(actual, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(pa) & np.isfinite(pb)
    a, pa, pb = a[ok], pa[ok], pb[ok]

    loss_a = np.abs(a - pa) ** power
    loss_b = np.abs(a - pb) ** power
    d = loss_a - loss_b
    n = d.size
    if n < 10:
        raise ValueError(f"Too few paired observations: {n}")

    var = newey_west_variance(d, max_lag=max_lag)
    stat = float(d.mean() / math.sqrt(var / n))
    used_lag = max_lag if max_lag is not None else int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    if harvey_correction:
        h = used_lag + 1
        adj = (n + 1 - 2 * h + h * (h - 1) / n) / n
        stat *= math.sqrt(max(adj, 1e-12))

    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return {
        "dm_stat": stat,
        "p_value": p_value,
        "mean_loss_diff": float(d.mean()),
        "n_pairs": int(n),
        "hac_lag": int(used_lag),
        "favours": "A" if d.mean() < 0 else "B",
    }


def pairwise_dm_table(forecasts: pd.DataFrame, models: list[str] | None = None,
                      power: int = 1, origin_level: bool = True) -> pd.DataFrame:
    """Pairwise Diebold-Mariano tests on common timestamps, averaged to origin level."""
    models = models or sorted(forecasts["model"].unique())
    wide = forecasts.pivot_table(index=["origin", "h"], columns="model", values="prediction", aggfunc="first")
    actual = forecasts.pivot_table(index=["origin", "h"], columns="model", values="actual", aggfunc="first").iloc[:, 0]
    frame = wide.join(actual.rename("actual"))
    frame = frame.dropna(subset=["actual"] + [m for m in models if m in frame.columns])

    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if a not in frame.columns or b not in frame.columns:
                continue
            if origin_level:
                g = frame.reset_index().groupby("origin")
                la = g.apply(lambda d: np.mean(np.abs(d["actual"] - d[a]) ** power), include_groups=False)
                lb = g.apply(lambda d: np.mean(np.abs(d["actual"] - d[b]) ** power), include_groups=False)
                d = (la - lb).to_numpy()
                n = d.size
                var = newey_west_variance(d)
                lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
                stat = float(d.mean() / math.sqrt(var / n))
                h = lag + 1
                stat *= math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
                p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
                res = {
                    "dm_stat": stat,
                    "p_value": p,
                    "mean_loss_diff": float(d.mean()),
                    "n_pairs": n,
                    "hac_lag": lag,
                    "favours": "A" if d.mean() < 0 else "B",
                }
            else:
                res = diebold_mariano(frame["actual"], frame[a], frame[b], power=power)
            rows.append({"model_A": a, "model_B": b, "winner": a if res["favours"] == "A" else b, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant_5pct"] = out["p_value"] < 0.05
        out["significant_1pct"] = out["p_value"] < 0.01
    return out

# ======================================================================================
# 7. ROLLING FORECAST & REPORT GENERATOR
# ======================================================================================
def _forecast_all(model: Any, y: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(model, "forecast_windows"):
        return model.forecast_windows(y, origins, horizon)
    point = np.full((len(origins), horizon), np.nan)
    pos = y.index.get_indexer(origins)
    values = y.to_numpy()
    for i, p in enumerate(pos):
        point[i] = model.forecast(pd.Series(values[: p + 1], index=y.index[: p + 1]), horizon)
    return point, None


def rolling_forecast(model: Any, y: pd.Series, origins: pd.DatetimeIndex, horizon: int = HORIZON,
                     model_name: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Generate long-format rolling forecasts welded strictly by timestamp."""
    name = model_name or getattr(model, "name", type(model).__name__)
    t0 = time.time()
    point, quantiles = _forecast_all(model, y, origins, horizon)
    elapsed = time.time() - t0

    n = len(origins)
    h_index = np.arange(1, horizon + 1)
    origin_col = np.repeat(np.asarray(origins), horizon)
    h_col = np.tile(h_index, n)
    target_times = pd.DatetimeIndex(origin_col) + pd.to_timedelta(h_col, unit="h")

    # Joined BY TIMESTAMP to guarantee zero off-by-one errors
    actual = y.reindex(target_times).to_numpy()

    out = pd.DataFrame({
        "model": name,
        "origin": origin_col,
        "h": h_col,
        "target_time": target_times,
        "actual": actual,
        "prediction": point.reshape(-1),
    })
    if quantiles is not None:
        for j, q in enumerate(QUANTILES):
            out[f"q{q:g}"] = quantiles[:, :, j].reshape(-1)

    if verbose:
        rate = n / elapsed if elapsed > 0 else float("inf")
        print(f"  {name:<28} {n:>5,} origins in {elapsed:>7.1f}s ({rate:>6.1f} origins/s)", flush=True)
    return out


def error_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    d = forecasts.dropna(subset=["actual", "prediction"]).copy()
    d["err"] = d["actual"] - d["prediction"]
    g = d.groupby(["model", "h"])
    return pd.DataFrame({
        "MAE": g["err"].apply(lambda e: float(np.mean(np.abs(e)))),
        "RMSE": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
    }).reset_index()


def summarise(forecasts: pd.DataFrame, scale: float, baseline_model: str | None = None) -> pd.DataFrame:
    rows = []
    for name, d in forecasts.groupby("model"):
        d = d.dropna(subset=["actual", "prediction"])
        row = {"model": name, "n_forecasts": int(len(d))}
        row.update(evaluate_forecast(d["actual"], d["prediction"], scale=scale))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    if baseline_model is not None and baseline_model in out.index:
        base_mae = out.loc[baseline_model, "MAE"]
        out["improvement_vs_baseline_%"] = (1.0 - out["MAE"] / base_mae) * 100.0
    return out.sort_values("MAE")


def build_all_result_tables(forecasts: pd.DataFrame, scale: float, baseline: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Compute and save all benchmark tables matching the original schema."""
    d_valid = forecasts.dropna(subset=["actual", "prediction"]).copy()

    # 1. Headline summary
    summary = summarise(forecasts, scale=scale, baseline_model=baseline)
    base_rmse = summary.loc[baseline, "RMSE"]
    summary["MAE_skill_%"] = summary["improvement_vs_baseline_%"]
    summary["RMSE_skill_%"] = (1.0 - summary["RMSE"] / base_rmse) * 100.0

    # 2. Detailed horizon table
    h_rows = []
    for (name, h), d in forecasts.groupby(["model", "h"]):
        d = d.dropna(subset=["actual", "prediction"])
        h_rows.append({
            "model": name, "h": int(h),
            "MAE": mae(d["actual"], d["prediction"]),
            "RMSE": rmse(d["actual"], d["prediction"]),
            "sMAPE": smape(d["actual"], d["prediction"]),
            "MASE": mase(d["actual"], d["prediction"], scale),
        })
    df_h = pd.DataFrame(h_rows)
    base_map = df_h[df_h["model"] == baseline].set_index("h")["MAE"]
    df_h["MAE_skill_%"] = (1.0 - df_h["MAE"] / df_h["h"].map(base_map)) * 100.0
    df_h = df_h.sort_values(["model", "h"])

    # 3. Probabilistic & Decile tables
    taus = np.asarray(QUANTILES, dtype=float)
    prob_rows = []
    cal_rows = []
    cov_rows = []
    diag_rows = []

    for name, d in forecasts.groupby("model"):
        has_q = all(c in d.columns for c in QUANTILE_COLS) and d[QUANTILE_COLS].notna().all(axis=None)
        if not has_q:
            continue
        diag_rows.append({"model": name, **count_quantile_crossings(d[QUANTILE_COLS].to_numpy(float))})

        # Rearrange before computing distributional metrics
        qmat = rearrange_quantiles(d[QUANTILE_COLS].to_numpy(dtype=float))
        actual_vals = d["actual"].to_numpy(dtype=float)

        prob_row = {
            "model": name,
            "CRPS": crps_from_quantiles(actual_vals, qmat, taus),
            "pinball_mean": float(np.mean([pinball_loss(actual_vals, qmat[:, j], float(t)) for j, t in enumerate(taus)])),
        }
        col_of = {level: j for j, level in enumerate(QUANTILES)}
        for lo_q, hi_q, level in INTERVAL_LEVELS:
            lo = qmat[:, col_of[lo_q]]
            hi = qmat[:, col_of[hi_q]]
            pct = int(round(level * 100))
            prob_row[f"PICP_{pct}"] = picp(actual_vals, lo, hi)
            prob_row[f"width_{pct}"] = interval_width(lo, hi)
            prob_row[f"winkler_{pct}"] = winkler_score(actual_vals, lo, hi, alpha=1.0 - level)
        prob_rows.append(prob_row)

        # Calibration
        emp = quantile_calibration(actual_vals, qmat, taus)
        for tau, observed in emp.items():
            cal_rows.append({"model": name, "nominal": tau, "empirical": observed, "deviation": observed - tau})

        # Coverage by horizon
        for (m_name, h_val), sub_d in d.groupby(["model", "h"]):
            sub_q = rearrange_quantiles(sub_d[QUANTILE_COLS].to_numpy(dtype=float))
            lo, hi = sub_q[:, 0], sub_q[:, -1]
            cov_rows.append({"model": m_name, "h": int(h_val), "PICP_80": picp(sub_d["actual"], lo, hi), "width_80": interval_width(lo, hi)})

    df_prob = pd.DataFrame(prob_rows).set_index("model") if prob_rows else pd.DataFrame()
    df_cal = pd.DataFrame(cal_rows)
    df_cov = pd.DataFrame(cov_rows)
    df_diag = pd.DataFrame(diag_rows).set_index("model") if diag_rows else pd.DataFrame()

    # 4. Metrics by month
    d_m = d_valid.copy()
    d_m["month"] = pd.DatetimeIndex(d_m["target_time"]).tz_localize(None).to_period("M").astype(str)
    d_m["abs_err"] = (d_m["actual"] - d_m["prediction"]).abs()
    d_m["err"] = d_m["actual"] - d_m["prediction"]
    g_m = d_m.groupby(["model", "month"])
    df_month = pd.DataFrame({"MAE": g_m["abs_err"].mean(), "bias": g_m["err"].mean(), "n": g_m["err"].size()}).reset_index()

    # 5. Peak metrics
    peak_rows = []
    for name, sub in d_valid.groupby("model"):
        g = sub.groupby("origin")
        peak_actual = g["actual"].max()
        peak_pred = g["prediction"].max()
        hour_actual = g.apply(lambda x: x.loc[x["actual"].idxmax(), "h"], include_groups=False)
        hour_pred = g.apply(lambda x: x.loc[x["prediction"].idxmax(), "h"], include_groups=False)
        hour_err = (hour_pred - hour_actual).abs()
        peak_rows.append({
            "model": name,
            "peak_MAE": float((peak_pred - peak_actual).abs().mean()),
            "peak_bias": float((peak_actual - peak_pred).mean()),
            "peak_MAPE_%": float(((peak_pred - peak_actual).abs() / peak_actual).mean() * 100),
            "peak_hour_exact_%": float((hour_err == 0).mean() * 100),
            "peak_hour_within_1h_%": float((hour_err <= 1).mean() * 100),
            "peak_hour_MAE_h": float(hour_err.mean()),
        })
    df_peak = pd.DataFrame(peak_rows).set_index("model").sort_values("peak_MAE")

    # 6. Error by time of day
    d_tod = d_valid.copy()
    d_tod["target_hour"] = pd.DatetimeIndex(d_tod["target_time"]).hour
    d_tod["abs_err"] = (d_tod["actual"] - d_tod["prediction"]).abs()
    df_tod = d_tod.groupby(["model", "target_hour"])["abs_err"].mean().reset_index().rename(columns={"abs_err": "MAE"})

    # 7. Significance Diebold-Mariano
    df_dm = pairwise_dm_table(forecasts)

    tables = {
        "summary": summary[["n_forecasts", "MAE", "RMSE", "sMAPE", "MAPE", "MASE", "improvement_vs_baseline_%"]],
        "summary_detailed": summary,
        "error_by_horizon": error_by_horizon(forecasts),
        "error_by_horizon_detailed": df_h,
        "probabilistic": df_prob,
        "calibration": df_cal,
        "coverage_by_horizon": df_cov,
        "metrics_by_month": df_month,
        "peak_metrics": df_peak,
        "error_by_time_of_day": df_tod,
        "significance_dm": df_dm,
        "quantile_crossing": df_diag,
    }

    for name, df in tables.items():
        if df is not None and not df.empty:
            path = RESULTS_DIR / f"{name}.csv"
            df.to_csv(path, index=not isinstance(df.index, pd.RangeIndex))
            if verbose:
                print(f"  Wrote {path.name:<32} ({len(df)} rows)")

    return tables

# ======================================================================================
# 8. EXPLORATORY DATA ANALYSIS (EDA)
# ======================================================================================
def build_eda_figures(y: pd.Series, save_dir: pathlib.Path = EDA_DIR) -> None:
    """Generate the 8 interactive EDA figures from the original repository."""
    import plotly.express as px
    import plotly.graph_objects as go

    save_dir.mkdir(parents=True, exist_ok=True)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    layout = dict(template="plotly_white", height=420, margin=dict(t=60))

    # 1. Full period
    daily = y.resample("D").mean()
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=y.index, y=y.to_numpy(), name="Hourly", line=dict(color="rgba(99,110,250,0.25)", width=0.5)))
    f1.add_trace(go.Scatter(x=daily.index, y=daily.to_numpy(), name="Daily mean", line=dict(color="#EF553B", width=1.6)))
    f1.update_layout(title="All-India demand, full period", yaxis_title="Demand (MW)", **layout)
    f1.write_html(save_dir / "01_full_period.html", include_plotlyjs="cdn")

    # 2. Hour of day
    g_h = y.groupby(y.index.hour)
    mean_h, q1_h, q3_h = g_h.mean(), g_h.quantile(0.25), g_h.quantile(0.75)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=q3_h.index, y=q3_h.to_numpy(), line=dict(width=0), showlegend=False))
    f2.add_trace(go.Scatter(x=q1_h.index, y=q1_h.to_numpy(), fill="tonexty", name="IQR", fillcolor="rgba(99,110,250,0.20)", line=dict(width=0)))
    f2.add_trace(go.Scatter(x=mean_h.index, y=mean_h.to_numpy(), name="Mean", line=dict(color="#636EFA", width=3)))
    f2.update_layout(title="Mean demand by hour of day (IST)", xaxis_title="Hour", yaxis_title="Demand (MW)", **layout)
    f2.write_html(save_dir / "02_hour_of_day.html", include_plotlyjs="cdn")

    # 3. Day of week
    g_dow = y.groupby(y.index.dayofweek).mean()
    f3 = go.Figure(go.Bar(x=day_names, y=g_dow.to_numpy(), marker_color="#636EFA"))
    f3.update_layout(title="Mean demand by day of week", yaxis_title="Demand (MW)", **layout)
    f3.update_yaxes(range=[g_dow.min() * 0.95, g_dow.max() * 1.02])
    f3.write_html(save_dir / "03_day_of_week.html", include_plotlyjs="cdn")

    # 4. Weekly profile
    how = y.index.dayofweek * 24 + y.index.hour
    prof = y.groupby(how).mean()
    f4 = go.Figure(go.Scatter(x=prof.index, y=prof.to_numpy(), mode="lines", line=dict(color="#636EFA", width=2)))
    f4.update_layout(title="Weekly seasonality (mean demand by hour of week)", xaxis_title="Hour of week (0 = Monday 00:00)", yaxis_title="Demand (MW)", **layout)
    f4.update_xaxes(tickmode="array", tickvals=list(range(0, 168, 24)), ticktext=day_names)
    f4.write_html(save_dir / "04_weekly_profile.html", include_plotlyjs="cdn")

    # 5. Monthly
    df_m = pd.DataFrame({"demand": y.to_numpy(), "year": y.index.year, "month": y.index.month})
    piv = df_m.groupby(["year", "month"])["demand"].mean().reset_index()
    f5 = px.line(piv, x="month", y="demand", color="year", markers=True)
    f5.update_layout(title="Monthly mean demand by year", xaxis_title="Month", yaxis_title="Demand (MW)", **layout)
    f5.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    f5.write_html(save_dir / "05_monthly.html", include_plotlyjs="cdn")

    # 6. Distribution
    f6 = go.Figure(go.Histogram(x=y.to_numpy(), nbinsx=80, marker_color="#636EFA"))
    f6.update_layout(title="Distribution of hourly demand", xaxis_title="Demand (MW)", yaxis_title="Hours", **layout)
    f6.write_html(save_dir / "06_distribution.html", include_plotlyjs="cdn")

    # 7. Missing map
    missing: list[str] = []
    if GAP_REPORT_PATH.exists():
        report = json.loads(GAP_REPORT_PATH.read_text(encoding="utf-8"))
        missing = report.get("missing_hours_list", [])
    counts = pd.Series(0, index=pd.date_range(y.index.min().tz_localize(None).normalize(), y.index.max().tz_localize(None).normalize(), freq="D"))
    if missing:
        m = pd.to_datetime(pd.Series(missing)).dt.normalize()
        vc = m.value_counts()
        counts.loc[counts.index.isin(vc.index)] = vc.reindex(counts.index[counts.index.isin(vc.index)]).to_numpy()
    f7 = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(), marker_color="#EF553B"))
    total_m = int(counts.sum())
    f7.update_layout(title=f"Missing hours before forward-fill ({total_m} of {len(y):,} = {total_m/len(y)*100:.3f}%)", yaxis_title="Missing hours that day", **layout)
    f7.write_html(save_dir / "07_missing_map.html", include_plotlyjs="cdn")

    # 8. Splits
    sp = chronological_splits(y)
    f8 = go.Figure()
    for name, colour in (("train", "#636EFA"), ("val", "#FFA15A"), ("test", "#EF553B")):
        idx = getattr(sp, name)
        seg = daily.loc[(daily.index >= idx[0]) & (daily.index <= idx[-1])]
        f8.add_trace(go.Scatter(x=seg.index, y=seg.to_numpy(), name=name, line=dict(color=colour, width=1.6)))
    f8.update_layout(title="Chronological splits (70 / 15 / 15) — never random", yaxis_title="Daily mean demand (MW)", **layout)
    f8.write_html(save_dir / "08_splits.html", include_plotlyjs="cdn")

    print(f"Generated 8 interactive EDA HTML figures in {save_dir}")

# ======================================================================================
# 9. EXPERIMENT EXECUTION STAGES
# ======================================================================================
def _append_forecasts(frames: list[pd.DataFrame], path=FORECAST_PATH) -> None:
    new = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)
    print(f"  Cached -> {path} ({len(new):,} rows, models: {sorted(new['model'].unique())})")


def stage_baselines() -> None:
    """Stage 1: Seasonal naive baselines and frozen MASE scale."""
    print("\n=== STAGE: Baselines ===")
    y = load_processed()
    sp = chronological_splits(y)
    train_vals = y.loc[sp.train]

    scale_daily = seasonal_naive_scale(train_vals, SEASONAL_PERIOD_DAILY)
    scale_weekly = seasonal_naive_scale(train_vals, SEASONAL_PERIOD_WEEKLY)
    print(f"  In-sample train naive MAE: m=24 {scale_daily:,.1f} MW | m=168 {scale_weekly:,.1f} MW")

    val_origins = make_origins(y, start=sp.val[0], end=sp.val[-1], min_context=MIN_CONTEXT)
    val_frames = [
        rolling_forecast(daily_naive(), y, val_origins, model_name="seasonal_naive_daily"),
        rolling_forecast(weekly_naive(), y, val_origins, model_name="seasonal_naive_weekly"),
    ]
    val_mae = {f["model"].iloc[0]: mae(f.dropna()["actual"], f.dropna()["prediction"]) for f in val_frames}
    stronger = min(val_mae, key=val_mae.get)
    print(f"  Validation MAE: {({k: round(v, 1) for k, v in val_mae.items()})}")
    print(f"  -> Stronger baseline: {stronger}")

    weekly_diag = check_weekly_seasonality(train_vals)
    scale = scale_weekly if stronger == "seasonal_naive_weekly" else scale_daily

    test_origins = make_origins(y, start=sp.test[0], end=sp.test[-1], min_context=MIN_CONTEXT)
    print(f"  Test origins: {len(test_origins):,}")
    test_frames = [
        rolling_forecast(daily_naive(), y, test_origins, model_name="seasonal_naive_daily"),
        rolling_forecast(weekly_naive(), y, test_origins, model_name="seasonal_naive_weekly"),
    ]
    _append_forecasts(test_frames)

    meta = {
        "mase_scale": scale,
        "mase_scale_basis": stronger,
        "scale_daily": scale_daily,
        "scale_weekly": scale_weekly,
        "stronger_baseline": stronger,
        "validation_mae": val_mae,
        "weekly_seasonality_check": weekly_diag,
        "n_test_origins": int(len(test_origins)),
        "n_val_origins": int(len(val_origins)),
        "test_start": str(sp.test[0]),
        "test_end": str(sp.test[-1]),
        "min_context": MIN_CONTEXT,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "random_seed": RANDOM_SEED,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"  Frozen MASE scale = {scale:,.1f} MW (in-sample {stronger} on train)")


def stage_lightgbm() -> None:
    """Stage 2: LightGBM tuning, 24 point models + 216 quantile models, rearrangement."""
    print("\n=== STAGE: LightGBM ===")
    y = load_processed()
    sp = chronological_splits(y)

    print("  Grid searching on validation split ...")
    best_params, records = tune_lightgbm(y, sp.train, sp.val)
    TUNING_PATH.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    fit_index = sp.train.union(sp.val)
    test_origins = make_origins(y, start=sp.test[0], end=sp.test[-1], min_context=MIN_CONTEXT)

    print(f"  Fitting 24 horizon point models on {len(fit_index):,} origins ...")
    t0 = time.time()
    model = LightGBMForecaster(params=best_params).fit(y, fit_index)
    print(f"    Point models fit in {time.time()-t0:.1f}s")

    point = model.forecast_batch(y, test_origins)
    n = len(test_origins)
    h_col = np.tile(np.arange(1, HORIZON + 1), n)
    origin_col = np.repeat(np.asarray(test_origins), HORIZON)
    target_times = pd.DatetimeIndex(origin_col) + pd.to_timedelta(h_col, unit="h")

    frame = pd.DataFrame({
        "model": "lightgbm",
        "origin": origin_col,
        "h": h_col,
        "target_time": target_times,
        "actual": y.reindex(target_times).to_numpy(),
        "prediction": point.reshape(-1),
    })

    print(f"  Fitting {len(QUANTILES)} quantiles x {HORIZON} horizons = {len(QUANTILES)*HORIZON} quantile models ...")
    t0 = time.time()
    for q in QUANTILES:
        qm = LightGBMForecaster(params=best_params, quantile=q).fit(y, fit_index)
        frame[f"q{q:g}"] = qm.forecast_batch(y, test_origins).reshape(-1)
        print(f"    q{q:g} done ({time.time()-t0:.1f}s elapsed)", flush=True)

    # Monotonicity check and rearrangement
    before_crossing = count_quantile_crossings(frame[QUANTILE_COLS].to_numpy(float))
    frame[QUANTILE_COLS] = rearrange_quantiles(frame[QUANTILE_COLS].to_numpy(float))
    print(f"  Quantile crossings before rearrangement: {before_crossing['fraction_of_rows']*100:.1f}% of rows, "
          f"max inversion {before_crossing['max_inversion']:,.0f} MW -> rearranged to monotone")

    _append_forecasts([frame])


def stage_timesfm(contexts: tuple[int, ...] = (896, 720)) -> None:
    """Stage 3: TimesFM 2.5 (200M) zero-shot evaluation for context 896 and 720."""
    print("\n=== STAGE: TimesFM 2.5 (200M) Zero-Shot ===")
    y = load_processed()
    sp = chronological_splits(y)
    test_origins = make_origins(y, start=sp.test[0], end=sp.test[-1], min_context=MIN_CONTEXT)

    frames = []
    for ctx in contexts:
        print(f"  Running TimesFM with context {ctx} on {len(test_origins):,} test origins ...")
        m = TimesFMForecaster(context_length=ctx)
        frames.append(rolling_forecast(m, y, test_origins, model_name=f"timesfm_ctx{ctx}"))
    _append_forecasts(frames)


def stage_summary() -> None:
    """Stage 4: Compute all metrics, statistical tests, and report tables."""
    print("\n=== STAGE: Summary & Evaluation Report ===")
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    scale = float(meta.get("mase_scale", 4445.199642420783))
    baseline = meta.get("stronger_baseline", "seasonal_naive_daily")

    fc = pd.read_parquet(FORECAST_PATH)
    tables = build_all_result_tables(fc, scale=scale, baseline=baseline)

    pd.set_option("display.width", 240)
    print(f"\nMASE scale {scale:,.1f} MW (in-sample {baseline} on train); skill quoted vs {baseline}\n")
    print(tables["summary_detailed"].round(3).to_string())

    if not tables["significance_dm"].empty:
        print("\n--- Diebold-Mariano Tests (Origin-level absolute loss, Newey-West HAC) ---")
        cols = ["model_A", "model_B", "winner", "dm_stat", "p_value", "significant_1pct"]
        print(tables["significance_dm"][cols].round(4).to_string(index=False))

    print("\n--- Peak-Demand Accuracy ---")
    print(tables["peak_metrics"].round(2).to_string())


def compare_with_reference() -> None:
    """Side-by-side comparison of current results vs original reference benchmarks."""
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON: OG REFERENCE vs NEW IMPLEMENTATION")
    print("=" * 80)

    ref_summary_path = REFERENCE_RESULTS_DIR / "summary_detailed.csv"
    new_summary_path = RESULTS_DIR / "summary_detailed.csv"

    if not ref_summary_path.exists():
        print(f"Reference results not found at {ref_summary_path}")
        return
    if not new_summary_path.exists():
        print(f"New results not found at {new_summary_path}. Run experiments first.")
        return

    ref_df = pd.read_csv(ref_summary_path).set_index("model")
    new_df = pd.read_csv(new_summary_path).set_index("model")

    metrics = ["MAE", "RMSE", "sMAPE", "MAPE", "MASE", "MAE_skill_%", "RMSE_skill_%"]
    common_models = [m for m in ref_df.index if m in new_df.index]

    comparison_rows = []
    for m in common_models:
        for metric in metrics:
            if metric in ref_df.columns and metric in new_df.columns:
                ref_val = ref_df.loc[m, metric]
                new_val = new_df.loc[m, metric]
                diff = new_val - ref_val
                pct_diff = (diff / ref_val) * 100.0 if ref_val != 0 else 0.0
                comparison_rows.append({
                    "Model": m,
                    "Metric": metric,
                    "OG Reference": round(ref_val, 3),
                    "New Executed": round(new_val, 3),
                    "Absolute Diff": round(diff, 3),
                    "% Diff": round(pct_diff, 2),
                })

    comp_df = pd.DataFrame(comparison_rows)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 160)
    print(comp_df.to_string(index=False))
    print("=" * 80)


STAGES = {
    "baselines": stage_baselines,
    "lightgbm": stage_lightgbm,
    "timesfm": stage_timesfm,
    "summary": stage_summary,
}

# ======================================================================================
# 10. CLI ENTRY POINT
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Indian Electricity Forecasting Pipeline")
    parser.add_argument("--download", action="store_true", help="Download raw Mendeley workbooks")
    parser.add_argument("--preprocess", action="store_true", help="Clean and aggregate raw workbooks to hourly Parquet")
    parser.add_argument("--eda", action="store_true", help="Generate interactive Plotly EDA HTML figures")
    parser.add_argument("--stage", choices=[*STAGES, "all"], help="Run pipeline stages")
    parser.add_argument("--compare-reference", action="store_true", help="Report OG reference vs new run results side-by-side")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    if args.download:
        download_data()
    if args.preprocess:
        preprocess_data()
    if args.eda:
        y = load_processed()
        build_eda_figures(y)
    if args.stage:
        names = list(STAGES) if args.stage == "all" else [args.stage]
        for name in names:
            STAGES[name]()
    if args.compare_reference:
        compare_with_reference()

    if not (args.download or args.preprocess or args.eda or args.stage or args.compare_reference):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
