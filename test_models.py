"""Offline sanity tests for the modular GARCH/VAR port.

Runnable with `python test_models.py` (no network): exercises the new evaluation
helpers and the forecast functions on synthetic data plus the cached price CSV.
"""
import os
import numpy as np
import pandas as pd

# Headless backend so importing the analysis modules never opens a window.
import matplotlib
matplotlib.use("Agg")

import evaluation as ev
from var_analysis import var_forecast, _build_diff, _select_lag
from garch_analysis import garch_forecast

PASSED = 0


def check(name, cond):
    global PASSED
    assert cond, f"FAILED: {name}"
    PASSED += 1
    print(f"  ok  {name}")


def synthetic_level(n=900, seed=0, start=5.0, sigma=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    steps = rng.normal(0, sigma, n)
    return pd.Series(start * np.exp(np.cumsum(steps)), index=idx)


def test_log_returns_and_realized_vol():
    level = synthetic_level()
    r = ev.log_returns(level)
    check("log_returns drops one observation", len(r) == len(level) - 1)

    # Constant return c% => realized vol = |c| * sqrt(252)
    const = pd.Series([0.5] * 300)
    expected = 0.5 * np.sqrt(ev.TRADING_DAYS)
    check("realized_vol of constant returns", abs(ev.realized_vol(const) - expected) < 1e-9)


def test_constant_vol_baseline_matches_realized():
    r = ev.log_returns(synthetic_level())
    check("constant_vol_forecast == realized_vol(train)",
          abs(ev.constant_vol_forecast(r, 60) - ev.realized_vol(r)) < 1e-12)


def test_vol_cv_mechanics():
    r = ev.log_returns(synthetic_level(n=700))
    cv = ev.rolling_origin_vol_cv(
        ev.constant_vol_forecast, r, initial=200, period=100, horizon=60, label="Const Vol"
    )
    check("vol CV returns folds", cv is not None and cv["n_folds"] >= 2)
    check("vol CV metrics finite and non-negative",
          np.isfinite(cv["mae"]) and np.isfinite(cv["rmse"]) and cv["mae"] >= 0)


def test_level_cv_unchanged():
    level = synthetic_level()
    cv = ev.rolling_origin_cv(
        ev.naive_rw_forecast, level, initial=200, period=100, horizon=60, label="Naive RW"
    )
    check("level CV still works for RW baseline", cv is not None and cv["n_folds"] >= 2)


def test_var_forecast_shape_and_inversion():
    fx = synthetic_level(n=600, seed=1)
    # A SELIC-like series aligned to fx (slow-moving rate around 12% p.a.)
    selic = pd.Series(12 + np.cumsum(np.random.default_rng(2).normal(0, 0.01, len(fx))),
                      index=fx.index)

    train = fx.iloc[:500]
    future_index = fx.index[500:560]
    out = var_forecast(train, future_index, selic)

    check("var_forecast length matches horizon", len(out) == len(future_index))
    check("var_forecast all positive (a price level)", np.all(np.asarray(out) > 0))
    check("var_forecast finite", np.all(np.isfinite(np.asarray(out))))
    # First forecast must be a continuous step from the last training level,
    # i.e. last_level * exp(small first diff) — within a few percent for FX.
    rel = abs(out[0] - train.iloc[-1]) / train.iloc[-1]
    check("var_forecast first step is continuous with history", rel < 0.1)


def test_var_helpers():
    fx = synthetic_level(n=400, seed=3)
    selic = pd.Series(12 + np.cumsum(np.random.default_rng(4).normal(0, 0.01, len(fx))),
                      index=fx.index)
    df_diff = _build_diff(fx, selic)
    check("_build_diff has both columns", list(df_diff.columns) == ["usd_brl_diff", "rate_diff"])
    check("_build_diff has no NaN", not df_diff.isna().any().any())
    check("_select_lag floored at >= 1", _select_lag(df_diff) >= 1)


def test_diebold_mariano():
    rng = np.random.default_rng(7)
    idx = pd.MultiIndex.from_tuples(
        [(c, c + j) for c in range(0, 600, 60) for j in range(60)],
        names=["cutoff", "date"],
    )
    bench = pd.Series(rng.normal(0, 1.0, len(idx)), index=idx)

    # Identical errors => no difference: stat ~ 0, p ~ 1.
    dm_same = ev.diebold_mariano(bench, bench.copy(), horizon=60, loss="abs")
    check("DM identical errors => |stat| tiny", abs(dm_same["stat"]) < 1e-6)
    check("DM identical errors => p ~ 1", dm_same["p_value"] > 0.99)

    # Model with uniformly smaller errors must win significantly (stat < 0).
    model = bench * 0.25
    dm = ev.diebold_mariano(model, bench, horizon=60, loss="abs")
    check("DM better model => negative stat", dm["stat"] < 0)
    check("DM better model => significant", dm["p_value"] < 0.05)
    check("DM better model => negative mean_diff", dm["mean_diff"] < 0)

    # No overlap in the index => None.
    other = pd.Series([1.0, 2.0, 3.0], index=pd.MultiIndex.from_tuples(
        [(9999, 0), (9999, 1), (9999, 2)], names=["cutoff", "date"]))
    check("DM with no common index => None",
          ev.diebold_mariano(model, other, horizon=60) is None)


def test_parse_fred_csv():
    from main import parse_fred_csv

    # Current FRED header is 'observation_date'; a '.' marks a missing day.
    text = "observation_date,DFF\n2024-01-02,5.33\n2024-01-03,.\n2024-01-04,5.31\n"
    df = parse_fred_csv(text)
    check("parse_fred_csv renames to data/valor", list(df.columns) == ["data", "valor"])
    check("parse_fred_csv drops '.' missing rows", len(df) == 2)
    check("parse_fred_csv parses dates", str(df["data"].dtype).startswith("datetime"))
    check("parse_fred_csv parses values", abs(df["valor"].iloc[0] - 5.33) < 1e-9)

    # Older exports use 'DATE' as the header — must still work (parser is
    # position-based, not name-based).
    legacy = "DATE,DFF\n2024-01-02,5.33\n"
    check("parse_fred_csv handles legacy DATE header",
          list(parse_fred_csv(legacy).columns) == ["data", "valor"])


def test_garch_forecast_on_real_data():
    csv = os.path.join("data", "usd_brl_history.csv")
    if not os.path.exists(csv):
        print("  skip  garch_forecast (no cached data/usd_brl_history.csv)")
        return
    df = pd.read_csv(csv, parse_dates=["data"]).set_index("data")
    returns = ev.log_returns(df["valor"])
    vol = garch_forecast(returns, 60)
    check("garch_forecast finite & positive", np.isfinite(vol) and vol > 0)
    # FX annualized vol realistically lands in a broad but bounded band.
    check("garch_forecast in a plausible range (1%-100%)", 1.0 < vol < 100.0)


if __name__ == "__main__":
    for fn in (
        test_log_returns_and_realized_vol,
        test_constant_vol_baseline_matches_realized,
        test_vol_cv_mechanics,
        test_level_cv_unchanged,
        test_var_forecast_shape_and_inversion,
        test_var_helpers,
        test_diebold_mariano,
        test_parse_fred_csv,
        test_garch_forecast_on_real_data,
    ):
        print(f"\n{fn.__name__}:")
        fn()
    print(f"\nAll assertions passed ({PASSED} checks).")
