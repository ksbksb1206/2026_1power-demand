import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = "data/processed/seoul_power_weather_hourly_2023-05-24_2026-05-24.csv"
DEFAULT_MODEL_OUTPUT = "models/lightgbm_power_demand.pkl"


BASE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "discomfort_index",
    "cooling_degree",
    "heating_degree",
    "daily_max_temperature",
    "daily_min_temperature",
    "daily_mean_temperature",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "season",
]


LAG_FEATURES = [
    "demand_lag_1h",
    "demand_lag_2h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_roll_mean_24h",
    "demand_roll_mean_168h",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model for hourly power demand prediction."
    )
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Input dataset CSV.")
    parser.add_argument(
        "--target",
        default="current_demand_mw",
        help="Target column to predict.",
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_OUTPUT,
        help="Output path for the trained model bundle.",
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/lightgbm_test_predictions.csv",
        help="Output CSV path for test predictions.",
    )
    parser.add_argument(
        "--importance-output",
        default="outputs/lightgbm_feature_importance.csv",
        help="Output CSV path for feature importance.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Final time-ordered ratio used as test data.",
    )
    parser.add_argument(
        "--no-lag",
        action="store_true",
        help="Train without demand lag/rolling features.",
    )
    return parser.parse_args()


def load_dataset(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_lag_features(df, target):
    df = df.copy()
    df["demand_lag_1h"] = df[target].shift(1)
    df["demand_lag_2h"] = df[target].shift(2)
    df["demand_lag_24h"] = df[target].shift(24)
    df["demand_lag_168h"] = df[target].shift(168)
    df["demand_roll_mean_24h"] = df[target].shift(1).rolling(24).mean()
    df["demand_roll_mean_168h"] = df[target].shift(1).rolling(168).mean()
    return df


def add_cyclical_time_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_training_frame(df, target, use_lag=True):
    if use_lag:
        df = add_lag_features(df, target)
    else:
        df = df.copy()
    df = add_cyclical_time_features(df)

    features = BASE_FEATURES + (LAG_FEATURES if use_lag else []) + [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    return df, features


def time_ordered_split(df, test_ratio):
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2,
    }


def train_model(train_df, features, target):
    model = LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[features],
        train_df[target],
        categorical_feature=["hour", "day_of_week", "month", "is_weekend", "season"],
    )
    return model


def save_outputs(model, train_df, test_df, features, target, args):
    pred = model.predict(test_df[features])
    metrics = regression_metrics(test_df[target].to_numpy(), pred)

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "metrics": metrics,
            "train_start": str(train_df["datetime"].min()),
            "train_end": str(train_df["datetime"].max()),
            "test_start": str(test_df["datetime"].min()),
            "test_end": str(test_df["datetime"].max()),
        },
        model_output,
    )

    predictions = test_df[["datetime", target]].copy()
    predictions["prediction"] = pred
    predictions["error"] = predictions[target] - predictions["prediction"]
    predictions["absolute_percentage_error"] = (
        predictions["error"].abs() / predictions[target].abs() * 100
    )
    predictions = predictions[
        [
            "datetime",
            target,
            "prediction",
            "error",
            "absolute_percentage_error",
        ]
    ]
    predictions_output = Path(args.predictions_output)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_output = Path(args.importance_output)
    importance_output.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_output, index=False, encoding="utf-8-sig")

    return metrics, model_output, predictions_output, importance_output


def main():
    args = parse_args()
    df = load_dataset(args.data)
    train_frame, features = make_training_frame(df, args.target, use_lag=not args.no_lag)
    train_df, test_df = time_ordered_split(train_frame, args.test_ratio)

    model = train_model(train_df, features, args.target)
    metrics, model_output, predictions_output, importance_output = save_outputs(
        model,
        train_df,
        test_df,
        features,
        args.target,
        args,
    )

    print("LightGBM training complete")
    print(f"rows train={len(train_df)} test={len(test_df)}")
    print(f"train period {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
    print(f"test period  {test_df['datetime'].min()} -> {test_df['datetime'].max()}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"model: {model_output}")
    print(f"predictions: {predictions_output}")
    print(f"feature importance: {importance_output}")


if __name__ == "__main__":
    main()
