import numpy as np
from numpy.typing import NDArray


def predict_classes(
    predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(predictions >= 0, 1.0, -1.0)


def calculate_metrics(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> dict[str, float]:
    predicted_classes = predict_classes(predictions)

    accuracy = float(np.mean(targets == predicted_classes))
    classification_error = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "classification_error": classification_error,
    }
