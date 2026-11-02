import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from sklearn.linear_model import LinearRegression


DATA_PATH = Path(__file__).with_name("gsl_monthly_master.csv")
OUTPUT_PATH = Path(__file__).with_name("gradient_matching_test.png")
INTERACTIVE_DATA_PATH = Path(__file__).with_name("gradient_matching_interactive.json")
DEFAULT_TRAIN_END = pd.Timestamp("2010-12-31")
REQUIRED_COLUMNS = [
    "Date",
    "Lake_Elevation_Mean_ft",
    "Total_Inflow_MAF",
    "NOAA_Station_Mean_Precip_mm",
]


@dataclass
class GradientMatchingFit:
    train: pd.DataFrame
    test: pd.DataFrame
    h_hat: np.ndarray
    dh_dt_hat: np.ndarray
    dh_dt_train_pred: np.ndarray
    elevation_forecast: np.ndarray
    coefficients: np.ndarray
    intercept: float
    train_rmse: float
    train_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float | None


def stable_rmse(errors: np.ndarray) -> float:
    scale = float(np.max(np.abs(errors)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.sqrt(np.mean((errors / scale) ** 2)))


def stable_mae(errors: np.ndarray) -> float:
    scale = float(np.max(np.abs(errors)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.mean(np.abs(errors / scale)))


def stable_r2(actual: np.ndarray, errors: np.ndarray) -> float | None:
    residual_norm = float(np.hypot.reduce(errors))
    centered_actual = actual - actual.mean()
    actual_norm = float(np.hypot.reduce(centered_actual))
    if actual_norm == 0.0:
        return 1.0 if residual_norm == 0.0 else 0.0
    ratio = residual_norm / actual_norm
    if ratio > np.sqrt(np.finfo(float).max):
        return None
    score = 1.0 - ratio**2
    return float(score) if np.isfinite(score) else None


def load_data() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    subset = df[REQUIRED_COLUMNS].copy()
    value_columns = REQUIRED_COLUMNS[1:]
    subset[value_columns] = subset[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    subset = subset.dropna(subset=REQUIRED_COLUMNS).sort_values("Date")
    subset = subset.reset_index(drop=True)

    if subset["Date"].duplicated().any():
        raise ValueError("The input data contains duplicate dates.")
    available_months = subset["Date"].tolist()
    return subset, available_months


def fit_gradient_matching(
    subset: pd.DataFrame, train_end: pd.Timestamp
) -> GradientMatchingFit:
    train = subset[subset["Date"] <= train_end].reset_index(drop=True)
    test = subset[subset["Date"] > train_end].reset_index(drop=True)
    if len(train) < 5:
        raise ValueError("At least five complete training observations are required.")
    if test.empty:
        raise ValueError(
            f"At least one complete observation after {train_end.year} is required."
        )

    t_days = (
        (train["Date"] - train["Date"].iloc[0]).dt.total_seconds().to_numpy()
        / 86400.0
    )
    if np.any(np.diff(t_days) <= 0):
        raise ValueError("Dates must be strictly increasing.")

    h_obs = train["Lake_Elevation_Mean_ft"].to_numpy(dtype=float)
    days_in_month = train["Date"].dt.days_in_month.to_numpy(dtype=float)
    inflow_maf_per_day = (
        train["Total_Inflow_MAF"].to_numpy(dtype=float) / days_in_month
    )
    precip_mm_per_day = (
        train["NOAA_Station_Mean_Precip_mm"].to_numpy(dtype=float) / days_in_month
    )

    spline = make_smoothing_spline(t_days, h_obs, lam=None)
    h_hat = spline(t_days)
    dh_dt_hat = spline.derivative()(t_days)

    elevation_centered = h_hat - h_hat.mean()
    X = np.column_stack(
        [inflow_maf_per_day, precip_mm_per_day, -elevation_centered]
    )
    model = LinearRegression(fit_intercept=True)
    model.fit(X, dh_dt_hat)

    coefficients = model.coef_
    intercept = float(model.intercept_)
    dh_dt_train_pred = model.predict(X)
    train_rmse = float(np.sqrt(np.mean((dh_dt_hat - dh_dt_train_pred) ** 2)))

    test_days_in_month = test["Date"].dt.days_in_month.to_numpy(dtype=float)
    test_inflow_maf_per_day = (
        test["Total_Inflow_MAF"].to_numpy(dtype=float) / test_days_in_month
    )
    test_precip_mm_per_day = (
        test["NOAA_Station_Mean_Precip_mm"].to_numpy(dtype=float)
        / test_days_in_month
    )

    forecast_dates = pd.concat(
        [train["Date"].tail(1), test["Date"]], ignore_index=True
    ).to_numpy(dtype="datetime64[ns]")
    elapsed_days = (
        np.diff(forecast_dates).astype("timedelta64[ns]").astype(np.float64)
        / 86400000000000.0
    )
    elevation_forecast = np.empty(len(test) + 1, dtype=float)
    elevation_forecast[0] = train["Lake_Elevation_Mean_ft"].iloc[-1]
    elevation_center = h_hat.mean()

    for i, elapsed in enumerate(elapsed_days):
        forcing = (
            coefficients[0] * test_inflow_maf_per_day[i]
            + coefficients[1] * test_precip_mm_per_day[i]
            + intercept
        )
        previous_offset = elevation_forecast[i] - elevation_center
        if np.isclose(coefficients[2], 0.0):
            next_offset = previous_offset + forcing * elapsed
        else:
            decay = np.exp(-coefficients[2] * elapsed)
            next_offset = (
                previous_offset * decay
                + forcing / coefficients[2] * (1.0 - decay)
            )
        elevation_forecast[i + 1] = elevation_center + next_offset
        if not np.isfinite(elevation_forecast[i + 1]):
            raise ValueError("The fitted ODE produced a non-finite test forecast.")

    test_elevation = test["Lake_Elevation_Mean_ft"].to_numpy(dtype=float)
    test_errors = test_elevation - elevation_forecast[1:]
    return GradientMatchingFit(
        train=train,
        test=test,
        h_hat=h_hat,
        dh_dt_hat=dh_dt_hat,
        dh_dt_train_pred=dh_dt_train_pred,
        elevation_forecast=elevation_forecast,
        coefficients=coefficients,
        intercept=intercept,
        train_rmse=train_rmse,
        train_r2=float(model.score(X, dh_dt_hat)),
        test_rmse=stable_rmse(test_errors),
        test_mae=stable_mae(test_errors),
        test_r2=stable_r2(test_elevation, test_errors),
    )


def export_interactive_data(
    subset: pd.DataFrame, available_months: list[pd.Timestamp]
) -> None:
    dates = subset["Date"]
    fits = []
    for train_end in available_months:
        train_count = int((dates <= train_end).sum())
        if train_count < 5 or train_count == len(subset):
            continue

        fit = fit_gradient_matching(subset, train_end)
        fits.append(
            {
                "date": train_end.strftime("%Y-%m-%d"),
                "split_index": train_count,
                "coefficients": [
                    float(value) for value in fit.coefficients
                ],
                "intercept": fit.intercept,
                "train_rmse": fit.train_rmse,
                "train_r2": fit.train_r2,
                "test_rmse": fit.test_rmse,
                "test_mae": fit.test_mae,
                "test_r2": fit.test_r2,
                "training_derivative": fit.dh_dt_hat.tolist(),
                "fitted_derivative": fit.dh_dt_train_pred.tolist(),
                "test_forecast": fit.elevation_forecast[1:].tolist(),
            }
        )

    if not fits:
        raise ValueError(
            "No training months have enough complete data for fitting and testing."
        )

    payload = {
        "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
        "elevation_ft": subset["Lake_Elevation_Mean_ft"].tolist(),
        "fits": fits,
    }
    INTERACTIVE_DATA_PATH.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    print(f"Interactive plot data: {INTERACTIVE_DATA_PATH}")
    print(
        f"Available training months: {fits[0]['date']}–{fits[-1]['date']} "
        f"({len(fits)} monthly cutoffs)"
    )


def print_results(fit: GradientMatchingFit, train_end: pd.Timestamp) -> None:
    print("--- GSL Gradient Matching Results ---")
    print(f"Training observations through {train_end.date()}: {len(fit.train)}")
    print(f"Test observations after {train_end.year}: {len(fit.test)}")
    print(
        f"Training date range: {fit.train['Date'].iloc[0].date()} to "
        f"{fit.train['Date'].iloc[-1].date()}"
    )
    print(
        f"Test date range: {fit.test['Date'].iloc[0].date()} to "
        f"{fit.test['Date'].iloc[-1].date()}"
    )
    print("Assumed ODE: dh/dt = alpha*Q + beta*P - gamma*(h - mean(h)) + c")
    print("Time derivative units: ft/day")
    print(f"alpha (inflow; ft/MAF):          {fit.coefficients[0]:.6g}")
    print(f"beta (precipitation; ft/mm):     {fit.coefficients[1]:.6g}")
    print(f"gamma (elevation feedback; 1/day): {fit.coefficients[2]:.6g}")
    print(f"c (baseline; ft/day):            {fit.intercept:.6g}")
    print(f"Training derivative R^2:         {fit.train_r2:.4f}")
    print(f"Training derivative RMSE:        {fit.train_rmse:.6g} ft/day")
    print(f"Test elevation RMSE:             {fit.test_rmse:.4f} ft")
    print(f"Test elevation MAE:              {fit.test_mae:.4f} ft")
    test_r2 = "undefined" if fit.test_r2 is None else f"{fit.test_r2:.4f}"
    print(f"Test elevation R^2:              {test_r2}")
    print("Spline smoothing strength: selected by generalized cross-validation")
    print(f"Test comparison graph: {OUTPUT_PATH}")


def save_default_plot(fit: GradientMatchingFit) -> None:
    forecast_dates = pd.concat(
        [fit.train["Date"].tail(1), fit.test["Date"]], ignore_index=True
    )
    test_elevation = fit.test["Lake_Elevation_Mean_ft"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    axes[0].plot(
        forecast_dates,
        fit.elevation_forecast,
        color="tab:orange",
        linestyle="--",
        label="ODE forecast",
    )
    axes[0].plot(
        forecast_dates,
        np.concatenate(
            [[fit.train["Lake_Elevation_Mean_ft"].iloc[-1]], test_elevation]
        ),
        color="black",
        linewidth=1.2,
        label="Actual elevation",
    )
    axes[0].set_ylabel("Elevation (ft)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title(
        f"Great Salt Lake: Fit Through {DEFAULT_TRAIN_END.year}, "
        f"Test After {DEFAULT_TRAIN_END.year}"
    )
    axes[1].plot(
        fit.train["Date"],
        fit.dh_dt_hat,
        color="purple",
        label="Training spline derivative",
    )
    axes[1].plot(
        fit.train["Date"],
        fit.dh_dt_train_pred,
        color="orange",
        linestyle="--",
        label="Training fitted ODE derivative",
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Elevation rate (ft/day)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the GSL gradient-matching model.")
    parser.add_argument(
        "--export-interactive",
        action="store_true",
        help="Precompute year-specific fits for the interactive website plot.",
    )
    args = parser.parse_args()

    subset, available_months = load_data()
    if args.export_interactive:
        export_interactive_data(subset, available_months)
        return

    fit = fit_gradient_matching(subset, DEFAULT_TRAIN_END)
    print_results(fit, DEFAULT_TRAIN_END)
    save_default_plot(fit)


if __name__ == "__main__":
    main()
