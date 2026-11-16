"""Consolidated verification test suite for the Indian Electricity Forecasting pipeline.

Tests:
1. Alignment: Pins f(T+h) against y(T+h-m) on synthetic ramp y(t)=t to guarantee zero off-by-one errors.
2. Leakage: Verifies features at origin T are identical whether built from full series or series truncated at T.
3. Metric Hand Calculations: Verifies MAE, RMSE, sMAPE, MAPE, MASE, pinball, Winkler, CRPS against paper arithmetic.
4. HAC Variance & Monotonicity: Verifies Newey-West HAC variance properties and quantile rearrangement guarantees.
"""
from __future__ import annotations

import torch
import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import main


# ======================================================================================
# 1. ALIGNMENT TESTS
# ======================================================================================
def _ramp(n: int = 1000) -> pd.Series:
    """Synthetic series y(t) = t. Any off-by-k alignment produces a constant error k."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="demand")


def test_seasonal_naive_daily_returns_the_value_24h_before_each_target():
    y = _ramp()
    model = main.daily_naive()
    origin_pos = 500
    history = y.iloc[: origin_pos + 1]

    pred = model.forecast(history, horizon=24)
    expected = [y.iloc[origin_pos + h - 24] for h in range(1, 25)]
    np.testing.assert_array_equal(pred, np.array(expected, dtype=float))


def test_seasonal_naive_weekly_returns_the_value_168h_before_each_target():
    y = _ramp()
    model = main.weekly_naive()
    origin_pos = 500
    history = y.iloc[: origin_pos + 1]

    pred = model.forecast(history, horizon=24)
    expected = [y.iloc[origin_pos + h - 168] for h in range(1, 25)]
    np.testing.assert_array_equal(pred, np.array(expected, dtype=float))


def test_ramp_gives_constant_error_equal_to_the_seasonal_period():
    y = _ramp()
    origins = main.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.daily_naive(), y, origins, horizon=24, verbose=False)

    err = fc["actual"] - fc["prediction"]
    assert err.notna().all()
    np.testing.assert_allclose(err.to_numpy(), 24.0)


def test_weekly_naive_on_ramp_is_wrong_by_exactly_168():
    y = _ramp()
    origins = main.make_origins(y, min_context=300, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.weekly_naive(), y, origins, horizon=24, verbose=False)
    np.testing.assert_allclose((fc["actual"] - fc["prediction"]).to_numpy(), 168.0)


def test_target_time_is_origin_plus_h():
    y = _ramp()
    origins = main.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.daily_naive(), y, origins, horizon=24, verbose=False)

    expected = pd.DatetimeIndex(pd.DatetimeIndex(fc["origin"]) + pd.to_timedelta(fc["h"].to_numpy(), unit="h"))
    pd.testing.assert_index_equal(pd.DatetimeIndex(fc["target_time"]), expected, check_names=False)


def test_actual_column_is_looked_up_by_timestamp():
    y = _ramp()
    origins = main.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.daily_naive(), y, origins, horizon=24, verbose=False)

    expected = y.reindex(pd.DatetimeIndex(fc["target_time"])).to_numpy()
    np.testing.assert_array_equal(fc["actual"].to_numpy(), expected)


def test_forecast_never_reuses_the_origin_value_as_the_first_actual():
    y = _ramp()
    origins = main.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.daily_naive(), y, origins, horizon=24, verbose=False)

    first = fc[fc["h"] == 1]
    origin_values = y.reindex(pd.DatetimeIndex(first["origin"])).to_numpy()
    assert not np.allclose(first["actual"].to_numpy(), origin_values)
    np.testing.assert_array_equal(first["actual"].to_numpy(), origin_values + 1.0)


def test_origins_leave_room_for_a_full_horizon():
    y = _ramp(n=500)
    origins = main.make_origins(y, min_context=200, stride_hours=6, horizon=24)
    assert len(origins) > 0
    assert origins.max() + pd.Timedelta(hours=24) <= y.index.max()


def test_origins_have_enough_history():
    y = _ramp(n=500)
    min_context = 200
    origins = main.make_origins(y, min_context=min_context, stride_hours=6, horizon=24)
    assert origins.min() >= y.index[min_context - 1]


def test_error_by_horizon_is_flat_for_a_seasonal_naive_on_a_ramp():
    y = _ramp()
    origins = main.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = main.rolling_forecast(main.daily_naive(), y, origins, horizon=24, verbose=False)
    curve = main.error_by_horizon(fc)
    assert len(curve) == 24
    np.testing.assert_allclose(curve["MAE"].to_numpy(), 24.0)


# ======================================================================================
# 2. LEAKAGE TESTS
# ======================================================================================
def _series(n: int = 1200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    t = np.arange(n)
    y = 150_000 + 20_000 * np.sin(2 * np.pi * t / 24) + 8_000 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 1000, n)
    return pd.Series(y, index=idx, name="demand")


@pytest.mark.parametrize("offset", [400, 700, 1000])
def test_origin_features_do_not_depend_on_the_future(offset):
    y = _series()
    origin = y.index[offset]
    full_row = main.origin_features(y).loc[origin]
    truncated_row = main.origin_features(y.loc[:origin]).loc[origin]
    pd.testing.assert_series_equal(full_row, truncated_row, check_names=False)


def test_every_origin_feature_is_finite_and_identical_under_truncation():
    y = _series()
    full = main.origin_features(y)
    for offset in range(400, 1100, 97):
        origin = y.index[offset]
        trunc = main.origin_features(y.loc[:origin]).loc[origin]
        assert np.isfinite(trunc.to_numpy()).all(), f"non-finite features at {origin}"
        np.testing.assert_array_equal(full.loc[origin].to_numpy(), trunc.to_numpy())


@pytest.mark.parametrize("horizon", [1, 12, 24])
def test_training_frame_features_precede_the_target(horizon):
    y = _series()
    X, target = main.training_frame(y, horizon)
    origin = X.index[len(X) // 2]
    trunc = main.features_at_origin(y.loc[:origin], horizon)
    for col in main.ORIGIN_FEATURES:
        assert X.loc[origin, col] == pytest.approx(trunc.iloc[0][col]), f"{col} differs"


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_training_frame_target_is_the_value_h_hours_after_the_origin(horizon):
    y = _series()
    X, target = main.training_frame(y, horizon)
    for origin in X.index[::250]:
        expected_time = origin + pd.Timedelta(hours=horizon)
        assert target.loc[origin] == pytest.approx(y.loc[expected_time])


def test_calendar_features_describe_the_target_not_the_origin():
    y = _series()
    X1, _ = main.training_frame(y, 1)
    origin = X1.index[500]
    assert X1.loc[origin, "hour"] == (origin + pd.Timedelta(hours=1)).hour

    X24, _ = main.training_frame(y, 24)
    assert X24.loc[origin, "hour"] == origin.hour


def test_rolling_std_is_trailing_not_centred():
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    values = np.concatenate([np.full(200, 100.0), np.full(n - 200, 500.0)])
    y = pd.Series(values, index=idx)

    feats = main.origin_features(y)
    last_flat = idx[199]
    assert feats.loc[last_flat, "roll_std_24"] == pytest.approx(0.0)
    assert feats.loc[last_flat, "roll_mean_24"] == pytest.approx(100.0)


# ======================================================================================
# 3. METRICS TESTS (HAND CALCULATIONS)
# ======================================================================================
ACTUAL = [100.0, 200.0, 300.0, 400.0]
PRED = [110.0, 190.0, 330.0, 360.0]


def test_mae_matches_hand_calculation():
    assert main.mae(ACTUAL, PRED) == pytest.approx(22.5)


def test_rmse_matches_hand_calculation():
    assert main.rmse(ACTUAL, PRED) == pytest.approx(math.sqrt(675.0))


def test_mape_matches_hand_calculation():
    assert main.mape(ACTUAL, PRED) == pytest.approx(8.75)


def test_smape_matches_hand_calculation():
    expected = np.mean([200 * 10 / 210, 200 * 10 / 390, 200 * 30 / 630, 200 * 40 / 760])
    assert main.smape(ACTUAL, PRED) == pytest.approx(expected)


def test_seasonal_naive_scale_matches_hand_calculation():
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert main.seasonal_naive_scale(y, 2) == pytest.approx(20.0)


def test_mase_is_mae_divided_by_scale():
    assert main.mase(ACTUAL, PRED, 4.5) == pytest.approx(22.5 / 4.5)


def test_evaluate_forecast_returns_expected_bundle():
    out = main.evaluate_forecast(ACTUAL, PRED, scale=4.5)
    assert out["MAE"] == pytest.approx(22.5)
    assert out["MASE"] == pytest.approx(5.0)


def test_picp_counts_actuals_inside_interval():
    actual = [1.0, 5.0, 10.0, 20.0]
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [2.0, 4.0, 11.0, 21.0]
    assert main.picp(actual, lower, upper) == pytest.approx(0.75)


def test_pinball_loss_matches_hand_calculation():
    assert main.pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    assert main.pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.2)
    assert main.pinball_loss([10.0], [8.0], 0.5) == pytest.approx(1.0)


def test_interval_width():
    assert main.interval_width([1.0, 2.0], [3.0, 6.0]) == pytest.approx(3.0)


# ======================================================================================
# 4. PROBABILISTIC & HAC STATISTICAL TESTS
# ======================================================================================
def test_winkler_inside_interval_is_width():
    assert main.winkler_score([100.0], [90.0], [110.0], alpha=0.2) == pytest.approx(20.0)


def test_winkler_below_interval_adds_scaled_shortfall():
    assert main.winkler_score([80.0], [90.0], [110.0], alpha=0.2) == pytest.approx(120.0)


def test_winkler_above_interval_adds_scaled_excess():
    assert main.winkler_score([120.0], [90.0], [110.0], alpha=0.2) == pytest.approx(120.0)


def test_winkler_punishes_uselessly_wide_interval():
    actual = [100.0, 105.0, 95.0]
    tight_lo, tight_hi = [90.0] * 3, [110.0] * 3
    wide_lo, wide_hi = [0.0] * 3, [1000.0] * 3
    assert main.picp(actual, wide_lo, wide_hi) == pytest.approx(1.0)
    assert main.winkler_score(actual, tight_lo, tight_hi) < main.winkler_score(actual, wide_lo, wide_hi)


def test_crps_matches_twice_mean_pinball():
    rng = np.random.default_rng(0)
    actual = rng.normal(100, 10, 50)
    taus = np.array([0.1, 0.5, 0.9])
    q = np.column_stack([actual - 12, actual + 1, actual + 12])
    expected = 2.0 * np.mean([main.pinball_loss(actual, q[:, j], t) for j, t in enumerate(taus)])
    assert main.crps_from_quantiles(actual, q, taus) == pytest.approx(expected)


def test_rearrangement_makes_quantiles_monotone():
    crossed = np.array([[10.0, 5.0, 7.0], [1.0, 2.0, 3.0]])
    fixed = main.rearrange_quantiles(crossed)
    assert np.all(np.diff(fixed, axis=1) >= 0)
    np.testing.assert_array_equal(fixed[0], [5.0, 7.0, 10.0])


def test_rearrangement_weakly_improves_pinball_loss():
    rng = np.random.default_rng(11)
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    actual = rng.normal(0, 1, 3000)
    truth = sps.norm.ppf(taus)
    noisy = truth + rng.normal(0, 1.2, (3000, taus.size))
    fixed = main.rearrange_quantiles(noisy)

    before = np.mean([main.pinball_loss(actual, noisy[:, j], t) for j, t in enumerate(taus)])
    after = np.mean([main.pinball_loss(actual, fixed[:, j], t) for j, t in enumerate(taus)])
    assert after <= before


def test_newey_west_exceeds_naive_variance_under_positive_autocorrelation():
    rng = np.random.default_rng(7)
    n = 2000
    e = rng.normal(0, 1, n)
    d = np.array([e[i] + 0.9 * e[i - 1] if i else e[i] for i in range(n)])
    naive_var = float(np.var(d))
    assert main.newey_west_variance(d, max_lag=10) > naive_var


def test_newey_west_matches_naive_variance_for_white_noise():
    rng = np.random.default_rng(8)
    d = rng.normal(0, 1, 5000)
    assert main.newey_west_variance(d, max_lag=0) == pytest.approx(float(np.var(d)), rel=1e-9)


def test_dm_favours_better_model():
    rng = np.random.default_rng(4)
    actual = rng.normal(100, 10, 500)
    good = actual + rng.normal(0, 1, 500)
    bad = actual + rng.normal(0, 8, 500)
    res = main.diebold_mariano(actual, good, bad)
    assert res["favours"] == "A"
    assert res["dm_stat"] < 0
    assert res["p_value"] < 0.01
