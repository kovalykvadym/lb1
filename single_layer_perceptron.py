import time

import numpy as np
from numpy.typing import NDArray

from prepare_data import get_prepared_data


def predict_raw(
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
    outputs = predict_raw(features, weights, bias)
    return np.where(outputs >= 0, 1.0, -1.0)


def train_one_sample(
    features: NDArray[np.float64],
    target: float,
    weights: NDArray[np.float64],
    bias: float,
    learning_rate: float,
) -> tuple[NDArray[np.float64], float, float]:
    output = predict_raw(features, weights, bias)

    gradient_output = (output - target) * (1 - output**2)

    weight_gradients = gradient_output * features
    bias_gradient = gradient_output

    new_weights = weights - learning_rate * weight_gradients
    new_bias = bias - learning_rate * bias_gradient

    loss = 0.5 * (target - output) ** 2

    return new_weights, new_bias, loss


def train(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    learning_rate: float,
    epochs: int,
) -> tuple[NDArray[np.float64], float, list[float]]:
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for features, target in zip(x_train, y_train):
            weights, bias, loss = train_one_sample(
                features,
                target,
                weights,
                bias,
                learning_rate,
            )

            epoch_loss += loss

        average_loss = epoch_loss / len(x_train)
        loss_history.append(average_loss)

    return weights, bias, loss_history


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, scaler = get_prepared_data()

    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")

    learning_rate = 0.1
    epochs = 20

    start_time = time.perf_counter()

    trained_weights, trained_bias, loss_history = train(
        x_train,
        y_train,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    training_time = time.perf_counter() - start_time

    predictions = predict_class(
        x_test,
        trained_weights,
        trained_bias,
    )

    correct = np.sum(predictions == y_test)
    accuracy = correct / len(y_test) * 100

    print("\nTraining results:")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Initial average loss: {loss_history[0]:.6f}")
    print(f"Final average loss: {loss_history[-1]:.6f}")

    print("\nTesting results:")
    print(f"Correct predictions: {correct} / {len(y_test)}")
    print(f"Classification accuracy: {accuracy:.2f}%")
