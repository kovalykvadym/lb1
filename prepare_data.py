from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "data" / "raw" / "data_banknote_authentication.txt"

FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]
TARGET_NAME = "class"
COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = {0, 1}


def load_dataset(file_path: Path = DATASET_PATH) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at {file_path}.")

    try:
        return pd.read_csv(file_path, delimiter=",")
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Dataset file {file_path} does not contain a valid dataset."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Dataset file has an invalid CSV structure: {file_path}."
        ) from error


def convert_numeric_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(COLUMN_NAMES) - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}.")

    dataset = dataset.copy()

    for column in COLUMN_NAMES:
        try:
            dataset[column] = pd.to_numeric(dataset[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Column '{column}' contains non-numeric values."
            ) from error

    return dataset


def validate_data(dataset: pd.DataFrame) -> None:
    if dataset.empty:
        raise ValueError("The dataset is empty.")

    if dataset.columns.tolist() != COLUMN_NAMES:
        raise ValueError(
            f"Expected columns in this order: {COLUMN_NAMES}. "
            f"Received: {dataset.columns.tolist()}."
        )

    if dataset.isnull().values.any():
        missing_counts = dataset.isnull().sum()

        raise ValueError(
            f"Dataset contains missing values: \n"
            f"{missing_counts[missing_counts > 0]}."
        )

    numeric_values = dataset[COLUMN_NAMES].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Dataset contains infinite or not-finite values.")

    actual_labels = set(dataset[TARGET_NAME].unique())

    if actual_labels != CLASS_LABELS:
        raise ValueError(
            f"Expected classes {CLASS_LABELS}, " f"received: {actual_labels}."
        )


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = dataset[FEATURE_NAMES].copy()
    target = dataset[TARGET_NAME].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    return x_train, x_test, y_train, y_test


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=FEATURE_NAMES,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=FEATURE_NAMES,
        index=x_test.index,
    )

    return x_train_scaled, x_test_scaled, scaler


def main() -> None:
    dataset = load_dataset()
    dataset = convert_numeric_columns(dataset)
    validate_data(dataset)

    x_train, x_test, y_train, y_test = split_dataset(dataset)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    print("Dataset validation: passed")
    print(f"Dataset size: {dataset.shape}")
    print(f"Training features: {x_train_scaled.shape}")
    print(f"Testing features: {x_test_scaled.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Testing labels: {y_test.shape}")
    print(f"Classes: {sorted(dataset[TARGET_NAME].unique())}")
    print("\nFirst three standardized training samples:")
    print(x_train_scaled.head(3))


if __name__ == "__main__":
    main()
