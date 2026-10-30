import numpy as np
from numpy.typing import NDArray


def predict(
    features: NDArray[np.float64],
    weights: NDArray[np.float64],
    bias: float,
) -> NDArray[np.float64]:
    weighted_sum = features @ weights + bias
    return np.tanh(weighted_sum)


def predict_class(
    features: NDArray[np.float64],
    weights: NDArray[np.float64],
    bias: float,
) -> NDArray[np.float64]:
    outputs = predict(features, weights, bias)
    return np.where(outputs >= 0, 1.0, -1.0)


def train_one_sample(
    features: NDArray[np.float64],
    target: float,
    weights: NDArray[np.float64],
    bias: float,
    learning_rate: float,
) -> tuple[NDArray[np.float64], float, float]:
    output = predict(features, weights, bias)

    gradient_output = (output - target) * (1.0 - output**2)

    weight_gradients = gradient_output * features
    bias_gradient = float(gradient_output)

    new_weights = weights - learning_rate * weight_gradients
    new_bias = float(bias - learning_rate * bias_gradient)

    loss = 0.5 * float((target - output) ** 2)

    return new_weights, new_bias, loss


def train(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    learning_rate: float,
    epochs: int,
) -> tuple[NDArray[np.float64], float, list[float]]:
    weights = np.zeros(x_data.shape[1], dtype=np.float64)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0

        for features, target in zip(x_data, y_data):
            weights, bias, loss = train_one_sample(
                features,
                float(target),
                weights,
                bias,
                learning_rate,
            )

            epoch_loss += loss

        average_loss = epoch_loss / len(x_data)
        loss_history.append(average_loss)

    return weights, bias, loss_history
