# All-India Electricity Demand Forecasting: Foundation Models vs. Tuned Local Models

A rigorous, compact, and reproducible empirical benchmark evaluating **zero-shot foundation models (Google TimesFM 2.5 200M)** against a **tuned local gradient-boosted model (LightGBM direct multi-horizon)** and **seasonal-naive baselines** on 46 months of all-India national electricity demand.

Evaluated over **832 rolling forecast origins** on a contiguous 7-month held-out test split (**19,968 forecasts per model tier**), with Newey-West HAC Diebold-Mariano significance testing, full distributional scoring (CRPS, Winkler scores, decile calibration), and daily peak demand analysis.

> **Key Finding:**
> **Yes, zero-shot TimesFM 2.5 decisively outperforms both the seasonal baselines and the tuned local model.**
> - **TimesFM 2.5 (ctx 896)** reduces Mean Absolute Error (MAE) by **35.7%** against the stronger seasonal baseline (*p* < 0.0001).
> - **Tuned LightGBM** achieved only a 7.6% reduction over the daily naive baseline, which **is not statistically significant** (*p* = 0.126).
> - TimesFM achieved this **strictly zero-shot**: no training, no fine-tuning, no calendar features, no holiday tags, and no covariates.

---

## 1. Executive Summary & Benchmark Results

### 1.1 Headline Point Forecast Accuracy (Horizon = 24h)

Evaluated on held-out test split (**2024-11-28 00:00 → 2025-06-24 23:00 IST**). All models evaluated across identical 832 origins at a 6-hour rolling stride.

| Model Tier | Context | MAE (MW) | RMSE (MW) | sMAPE (%) | MAPE (%) | MASE | MAE Skill (%) | RMSE Skill (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TimesFM 2.5 (Zero-shot)** | **896** | **3,642.1** | **5,838.4** | **1.88%** | **1.89%** | **0.819** | **+35.7%** | **+32.2%** |
| TimesFM 2.5 (Zero-shot) | 720 | 3,719.5 | 5,893.6 | 1.92% | 1.93% | 0.837 | +34.3% | +31.6% |
| LightGBM (Tuned Direct) | 336 | 5,231.0 | 7,310.8 | 2.71% | 2.72% | 1.177 | +7.6% | +15.1% |
| Seasonal Naive ($m=24$) *(Baseline)* | 24 | 5,662.0 | 8,615.7 | 2.95% | 2.96% | 1.274 | — | — |
| Seasonal Naive ($m=168$) | 168 | 8,328.9 | 11,742.5 | 4.28% | 4.29% | 1.874 | −47.1% | −36.3% |

*Note on MASE*: Scaled by the in-sample MAE of the stronger daily seasonal naive on the **training split** (**4,445.2 MW**, $m=24$), frozen once and applied identically to every model tier. $\text{MASE} < 1.0$ indicates outperforming the in-sample naive baseline. Only TimesFM achieves $\text{MASE} < 1.0$.

---

### 1.2 Statistical Significance (Diebold-Mariano Tests)

Because 24-hour windows generated at a 6-hour stride overlap by 18 hours, consecutive forecast errors share target hours and are strongly autocorrelated. Treating observations as independent would drastically underestimate variance and manufacture false significance.

Significance is evaluated using the **Diebold-Mariano test on origin-level mean absolute error differentials**, with a **Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent)** spectral variance estimator (Bartlett kernel) and **Harvey-Leybourne-Newbold small-sample correction**:

| Comparison | Superior Model | DM Statistic | *p*-value | Significant at 1%? | Conclusion |
|---|:---:|:---:|:---:|:---:|---|
| **TimesFM (896) vs. LightGBM** | **TimesFM (896)** | **+9.011** | **< 0.0001** | ✅ Yes | Foundation model decisively superior |
| **TimesFM (896) vs. Naive (m=24)** | **TimesFM (896)** | **+8.883** | **< 0.0001** | ✅ Yes | Foundation model beats persistence |
| **TimesFM (896) vs. Naive (m=168)** | **TimesFM (896)** | **+10.168** | **< 0.0001** | ✅ Yes | Foundation model beats weekly naive |
| **LightGBM vs. Naive (m=24)** | LightGBM | −1.533 | **0.1257** | ❌ **No** | **Statistically indistinguishable from naive baseline** |
| LightGBM vs. Naive (m=168) | LightGBM | −5.703 | < 0.0001 | ✅ Yes | LightGBM beats weekly naive |
| Naive ($m=24$) vs. Naive ($m=168$) | Naive ($m=24$) | −4.829 | < 0.0001 | ✅ Yes | Daily naive beats weekly naive in India |
| TimesFM (896) vs. TimesFM (720) | TimesFM (896) | +1.986 | 0.0473 | ❌ (5% only) | Longer context yields marginal gain |

> **Critical Takeaway**: A heavily engineered LightGBM model utilizing 8 lags, 4 rolling statistics, differences, holiday calendars, and cyclic encodings **cannot be statistically distinguished from "yesterday, same hour"** on this held-out test period. In contrast, TimesFM outperforms both by an overwhelming margin.

---

### 1.3 Probabilistic Scoring & Uncertainty Calibration

All models emit 9 deciles ($q_{0.1}, q_{0.2}, \dots, q_{0.9}$). Continuous Ranked Probability Score (CRPS) measures overall distributional quality; Winkler score measures the width of central prediction intervals plus a heavy penalty for misses.

| Model Tier | CRPS (MW) | Mean Pinball Loss | Winkler@80 | Width@80 (MW) | PICP@80 (Nominal 0.80) |
|---|:---:|:---:|:---:|:---:|:---:|
| **TimesFM 2.5 (ctx 896)** | **2,905.8** | **1,452.9** | **18,109.6** | 11,665.7 | **0.820 (82.0%)** |
| TimesFM 2.5 (ctx 720) | 2,971.3 | 1,485.7 | 18,507.6 | 11,891.0 | 0.809 (80.9%) |
| LightGBM (Tuned) | 4,100.4 | 2,050.2 | 26,941.4 | 10,637.6 | 0.583 (58.3%) |

#### Empirical Calibration Across Deciles
Comparison of nominal prediction interval coverage (PICP) against empirical reality:

| Interval Level | Nominal Target | TimesFM (896) | TimesFM (720) | LightGBM (Rearranged) |
|---|:---:|:---:|:---:|:---:|
| Central 20% ($P_{40} - P_{60}$) | 0.200 | **0.209** | 0.208 | 0.128 |
| Central 40% ($P_{30} - P_{70}$) | 0.400 | **0.413** | 0.412 | 0.259 |
| Central 60% ($P_{20} - P_{80}$) | 0.600 | **0.614** | 0.606 | 0.403 |
| Central 80% ($P_{10} - P_{90}$) | 0.800 | **0.820** | 0.809 | 0.583 |

**Quantile Crossing Correction**: Fitting 9 independent LightGBM quantile models produces monotonicity inversions ($\hat{q}_{0.4} > \hat{q}_{0.6}$) on **83% of rows** with violations up to **9,606 MW**. To ensure valid distributions, all LightGBM quantiles pass through row-wise rearrangement sorting (Chernozhukov et al., 2010), weakly reducing estimation error. TimesFM produces 0% crossing violations natively (`fix_quantile_crossing=True`).

---

### 1.4 Daily Peak Demand Accuracy

The daily peak demand determines generating reserve margins, peaking plant dispatch, and load-shedding avoidance.

| Model Tier | Peak MAE (MW) | Peak Bias (MW) | Peak MAPE (%) | Peak Hour Exact (%) | Peak Hour Within $\pm 1$h (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **TimesFM 2.5 (ctx 896)** | **3,735.6** | +631.1 | **1.71%** | **72.5%** | **87.4%** |
| TimesFM 2.5 (ctx 720) | 3,855.5 | +747.9 | 1.77% | 72.2% | 86.8% |
| Seasonal Naive ($m=24$) | 4,993.3 | **+75.0** | 2.31% | 65.3% | 82.9% |
| **LightGBM (Tuned)** | **6,000.4** | **+4,917.9** | **2.70%** | **44.5%** | **76.0%** |
| Seasonal Naive ($m=168$) | 8,346.4 | +534.0 | 3.81% | 60.0% | 78.4% |

*Bias sign*: $\text{Actual} - \text{Predicted}$ (positive = under-forecast).

> **Operational Vulnerability in Local Model**: LightGBM suffers from severe conditional mean shrinkage, systematically **under-forecasting the daily national peak by 4,918 MW** (~2.4% of total grid load) and identifying the correct peak hour only 44.5% of the time (substantially worse than the daily naive's 65.3%). In a real grid dispatch environment, shaving 5 GW off the daily peak introduces immediate blackout risks.

---

## 2. Why Electricity-Demand Forecasting Matters

Electricity cannot be stored cost-effectively at national scale; generation must match load second-by-second to maintain a 50 Hz nominal grid frequency:
1. **Operating Reserves & Unit Commitment**: Thermal power stations (coal, gas) require 4 to 12 hours of notice to fire up from cold or warm states. Day-ahead (24h) demand forecasts determine which plants are synchronized to the grid.
2. **Renewable Energy Integration**: India has installed over 180 GW of variable renewable energy (solar and wind). Accurate net-load forecasting prevents renewable curtailment and thermal cycling inefficiencies.
3. **Economic Dispatch & Power Exchanges**: Over 10% of India's daily electricity is traded on the Indian Energy Exchange (IEX). Accurate forecasts reduce exposure to deviation settlement mechanism (DSM) penalties.
4. **Preventing Cascading Grid Failures**: Under-forecasting capacity requirements risks under-frequency load shedding or catastrophic grid collapse (such as the July 2012 Northern Grid collapse affecting 620 million people).

---

## 3. Dataset Provenance, Ingestion & Preprocessing

- **Source**: Mendeley Data, DOI: [10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2), Version 2.
- **Authors**: Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2024).
- **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Original Source**: National Load Despatch Centre (NLDC), Grid-India (formerly POSOCO), Ministry of Power, Government of India.
- **Telemetry Point**: `NLDC_DEMAND|P` (All-India instantaneous demand in **Megawatts (MW)**).
- **Total Duration**: 2021-09-01 00:00 → 2025-06-24 23:00 IST (**46 contiguous months, 33,432 hourly observations**). Post-COVID throughout (no artificial lockdown collapse to adjust for).

### 3.1 Preprocessing Challenges & Solutions

| Challenge | Raw Data Characteristic | Pipeline Treatment | Methodological Rationale |
|---|---|---|---|
| **Heterogeneous Regimes** | 13 files at 5-min (Sep 21–Sep 22), 15 files at 10-s (Oct 22–Dec 23), 1 file pre-aggregated hourly (Jan 24–Jun 25). | Resample Regimes A and B using **hourly mean**. Verified against sampling interval in `_osi_config`. | A roughness test on boundary months (Dec 2023 vs Jan 2024) confirmed the depositor used hourly means. Using instantaneous sampling would induce a 1.5% structural break inside the training set. |
| **Excel Summary Rows** | Regime C has 6 trailing summary rows (Min, Max, Average, Sum). | Strict parsing with `ts.notna()`; dropped 6 non-timestamp rows. | Reading naively yields a reported national peak of **2.5 billion MW**. |
| **Ambiguous Date Strings** | Regime C timestamps are DD-MM-YYYY strings. | Enforced `pd.to_datetime(..., dayfirst=True)`. | Prevents pandas from silently swapping day and month for the first 12 days of each month. |
| **SCADA Telemetry Dropouts** | Instantaneous zero-dropouts (0.0 MW) during sensor disconnects. | Replaced exact 0.0 with `NaN` **before** temporal aggregation. | Leaving 0.0 unmasked drags hourly mean demand down by up to 2.3% (4,099 MW). |
| **Missing Observations** | 12 missing hours out of 33,432 (0.036%). | Reindexed to strict hourly grid; forward filled (`ffill()`) using past observations only. | Linear interpolation across gaps would look forward in time, causing data leakage. |
| **Timezone & DST** | Raw files carry no timezone offset. | Localized to `Asia/Kolkata` (IST, UTC+05:30). | India does not observe Daylight Saving Time; verified no 23-hour or 25-hour days exist. |

---

## 4. Evaluation Methodology: Leakage-Free Design

### 4.1 Chronological Splits (Never Random)
Random train/test splits in time-series allow models to interpolate between $t-1$ and $t+1$, destroying the forecasting task. The data is partitioned strictly chronologically:
- **Train Split (70%)**: `2021-09-01 00:00` → `2024-05-03 01:00` (23,402 hours)
- **Validation Split (15%)**: `2024-05-03 02:00` → `2024-11-27 17:00` (5,014 hours)
- **Test Split (15%)**: `2024-11-28 00:00` → `2025-06-24 23:00` (5,016 hours)

### 4.2 Rolling-Origin Evaluation
- **Forecasting Horizon**: $H = 24$ hours ahead.
- **Origin Stride**: $S = 6$ hours. Stepping by 6 hours ensures all four quarters of the diurnal cycle (00:00, 06:00, 12:00, 18:00) serve as forecast origins.
- **Identical Origins**: Fixed to 832 origins on the held-out test split, dictated by the largest warmup context required (896 hours). Every model tier is scored on the exact same 19,968 (origin, horizon) pairs.

---

## 5. Model Architectures & Parameterization

### 5.1 Primary Benchmark: Google TimesFM 2.5 (200M)
- **Variant**: `google/timesfm-2.5-200m-pytorch`, 200M parameters, decoder-only transformer architecture with patched time-series representations.
- **Mode**: Strictly **zero-shot**, univariate CPU inference.
- **Context Length**: **896 hours** (~37 days of trailing history).
- **Validation Context Sweep**:
  - $L=168$: Val MAE 4,513 MW
  - $L=336$: Val MAE 3,961 MW
  - $L=720$: Val MAE 3,701 MW
  - **$L=896$: Val MAE 3,568 MW (Optimal)**
  - $L=1024$: Val MAE 3,571 MW (Within 0.1% of 896, tied)
- **Quantile Output**: Emits point forecasts alongside 9 deciles ($q_{0.1} \dots q_{0.9}$) via a continuous quantile head with built-in crossing correction.

### 5.2 Local Model: Tuned LightGBM Direct Multi-Horizon
- **Strategy**: Direct multi-horizon forecasting ($h=1..24$ independent models for point predictions + $24 \times 9 = 216$ quantile models). Avoids recursive error accumulation.
- **Engineered Features (Origin-time only $y_{\le T}$)**:
  - Lags: 0, 1, 2, 3, 24, 48, 168, 336 hours.
  - Rolling statistics: Trailing 24h and 168h rolling mean and standard deviation.
  - Differences: 24h change ($y_T - y_{T-24}$) and 168h change ($y_T - y_{T-168}$).
  - Target-time calendar: Hour of day, day of week, month, day of year, weekend indicator, cyclical sine/cosine transforms.
  - Indian public holidays: Mapped via `holidays.India`.
- **Validation Tuning**: Grid search over tree capacity (`num_leaves`: 7, 15, 31, 63, 127) and learning rate. Optimal configuration: `num_leaves=15`, `learning_rate=0.05`, `n_estimators=400`, `min_child_samples=20`, `subsample=0.9`, `colsample_bytree=0.9`. Lower tree capacity prevents overfitting the smooth daily profile.

### 5.3 Baselines & The India-Specific Seasonality Discovery
- **Daily Seasonal Naive ($m=24$)**: $\hat{y}(T+h) = y(T+h-24)$
- **Weekly Seasonal Naive ($m=168$)**: $\hat{y}(T+h) = y(T+h-168)$

#### Why Weekly Naive Loses to Daily Naive in India
In Western grids, weekly naive strongly beats daily naive because weekends have significantly lower commercial/industrial load than weekdays.
In India, **the weekly naive loses dramatically** (8,328.9 MW vs. 5,661.9 MW, $p < 0.0001$):
- Day-of-week demand variation is only **4.41% of mean demand**, whereas hour-of-day diurnal variation is **20.29%**.
- National demand is anchored by continuous heavy industrial manufacturing, railway traction, and agricultural water pumping running 7 days a week.
- Only Sunday displays a minor dip (−3.40%). Monday through Saturday sit within 1.4% of each other.
- Consequently, carrying a forecast 7 days forward incurs 7 days of level drift that far outweighs weekday matching.

---

## 6. Architectural Comparison: Alternative Models Considered

| Model Class | Candidate Architecture | Why It Was Not Selected / Evaluated |
|---|---|---|
| **Classical Statistical** | **ARIMA / SARIMA** | Univariate linear assumption fails to capture complex dual diurnal-weekly seasonality. Fitting 832 rolling origins is computationally intractable; recursive multi-step forecasting causes rapid error compounding across 24 steps. |
| **Tree Ensembles** | **XGBoost / CatBoost** | Similar empirical performance to LightGBM. LightGBM was selected because its histogram-based binning executes 5–10× faster and provides direct quantile loss optimization for 216 models. |
| **Deep Learning** | **LSTM / GRU / RNN** | Require extensive hyperparameter tuning, sequence padding, and GPU training. Frequently underperform gradient-boosted trees on tabular time-series features without exogenous weather inputs. |
| **Deep Forecasting** | **Temporal Fusion Transformer (TFT)** | Highly parameter-heavy architecture intended for multi-series panel datasets with extensive static and time-varying covariates. Prone to severe overfitting on a single national load series. |
| **Deep Forecasting** | **N-BEATS / N-HiTS** | Pure univariate neural forecasters. Pretraining foundation models (TimesFM) have superseded custom per-series neural models on zero-shot generalization. |
| **Foundation Models** | **Amazon Chronos / Chronos-2** | T5-based language model tokenization of time-series. TimesFM 2.5 was chosen as the primary tier due to its GIFT-Eval benchmark leadership, native decile quantile heads, and high CPU inference throughput (~1.3 origins/s). |

---

## 7. Project Architecture: Compact & Unified Structure

Unlike fragmented implementations that disperse logic across 18+ small files, this project consolidates functionality into a cohesive, readable footprint:

```text
├── main.py              # Unified pipeline: Config, Ingestion, Preprocessing,
│                        # Splits, Features, Models, Evaluation, EDA & CLI
├── app.py               # Streamlit interactive dashboard & visualization app
├── test_pipeline.py     # Consolidated test suite (Leakage, Alignment, Metrics, HAC)
├── requirements.txt     # Pinned dependencies with verified Windows compatibility
├── README.md            # Comprehensive project documentation & benchmark analysis
├── data/
│   ├── raw/             # Raw Mendeley Excel workbooks
│   └── processed/       # demand_hourly.parquet & preprocessing_report.json
├── results/             # Benchmark tables (CSV), metadata, and tuning logs
└── reference_results/   # Original benchmark baseline values for validation
```

---

## 8. Installation & Execution Guide

### 8.1 Environment Setup
Tested on **Python 3.11** and **Python 3.12** (Windows 11 / Linux / macOS). Pure CPU execution—no GPU required.

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install verified dependencies
pip install -r requirements.txt
```

> **Windows Security Compatibility Note**: In Windows 11 with Smart App Control or AppLocker enabled, unsigned cutting-edge packages may trigger `OSError: [WinError 4551] An Application Control policy has blocked this file`. This project specifies `pandas==2.2.3` and `pyarrow==17.0.0`, which are verified stable releases recognized by Windows application security.

---

### 8.2 Reproducing the Full Pipeline

```powershell
# Step 1: Run verification test suite (42 tests in ~4s)
pytest test_pipeline.py -v

# Step 2: Download raw Mendeley workbooks (748 MB, ~10 min)
python main.py --download

# Step 3: Run data cleaning and preprocessing (~7 min)
python main.py --preprocess

# Step 4: Generate exploratory data analysis figures
python main.py --eda

# Step 5: Run reproducible model experiments
python main.py --stage baselines    # Evaluate Seasonal Naives
python main.py --stage lightgbm     # Tune & fit LightGBM point & quantile models
python main.py --stage timesfm      # Run TimesFM 2.5 zero-shot evaluation
python main.py --stage summary      # Generate all result tables & metrics

# Or run all stages sequentially:
python main.py --stage all

# Step 6: Verify executed results against reference benchmark
python main.py --compare-reference

# Step 7: Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## 9. Limitations & Future Work

1. **Univariate Setting (Absence of Weather Covariates)**:
   - Temperature and humidity are the primary exogenous drivers of cooling load in Indian summers.
   - The absence of weather forecasts penalizes LightGBM more than TimesFM, as gradient-boosted trees excel at exploiting non-linear temperature response functions.
2. **Zero-Shot Foundation Model**:
   - TimesFM was evaluated strictly out-of-the-box. Parameter-efficient fine-tuning (PEFT/LoRA) on Indian grid data represents a promising avenue for further error reduction.
3. **Single National Aggregate Series**:
   - National demand aggregates across 5 regional grids, smoothing localized load-shedding and agricultural feeder rationing. State-level disaggregation would test foundation models on noisier, peakier series.
4. **Asymmetric Loss Functions for Peak Sizing**:
   - Standard L1 loss penalizes over-forecasting and under-forecasting equally. Training tree ensembles with asymmetric pinball or custom reserve-margin loss functions could eliminate LightGBM's 4.9 GW peak under-prediction.

---

## 10. License & Attribution

- **Code License**: MIT License.
- **Dataset License**: CC BY 4.0 (Attribution required).
- **Data Citation**: Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2024), *Electricity Demand, Solar and Wind Generation Data (September 2021 – June 2025) of India at 1-hour interval*, Mendeley Data, V2, doi: [10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2).
- **Original Data Source**: Grid-India (formerly POSOCO) / NERLDC, Ministry of Power, Government of India.
