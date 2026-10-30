import numpy as np
from numpy.typing import NDArray


def activation(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(values)


def initialize_parameters() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(42)

    hidden_weights = rng.standard_normal((4, 4)) * 0.1
    hidden_bias = np.zeros(4, dtype=np.float64)

    output_weights = rng.standard_normal((4, 1)) * 0.1
    output_bias = np.zeros(1, dtype=np.float64)

    return hidden_weights, hidden_bias, output_weights, output_bias


def predict(
    features: NDArray[np.float64],
    hidden_weights: NDArray[np.float64],
    hidden_bias: NDArray[np.float64],
    output_weights: NDArray[np.float64],
    output_bias: NDArray[np.float64],
) -> NDArray[np.float64]:
    hidden_input = features @ hidden_weights + hidden_bias
    hidden_activation = activation(hidden_input)

    output_input = hidden_activation @ output_weights + output_bias
    predictions = activation(output_input)

    return predictions.ravel()


def train_one_sample(
    sample_features: NDArray[np.float64],
    sample_target: float,
    hidden_weights: NDArray[np.float64],
    hidden_bias: NDArray[np.float64],
    output_weights: NDArray[np.float64],
    output_bias: NDArray[np.float64],
    learning_rate: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    hidden_input = sample_features @ hidden_weights + hidden_bias
    hidden_activation = activation(hidden_input)

    output_input = hidden_activation @ output_weights + output_bias
    prediction = activation(output_input)[0]

    output_gradient = (prediction - sample_target) * (1.0 - prediction**2)
    hidden_gradient = (
        output_gradient * output_weights[:, 0] * (1.0 - hidden_activation**2)
    )

    output_weights -= learning_rate * output_gradient * hidden_activation.reshape(-1, 1)
    output_bias -= learning_rate * output_gradient

    hidden_weights -= learning_rate * np.outer(sample_features, hidden_gradient)
    hidden_bias -= learning_rate * hidden_gradient

    sample_loss = 0.5 * float((sample_target - prediction) ** 2)

    return hidden_weights, hidden_bias, output_weights, output_bias, sample_loss


def train(
    features_train: NDArray[np.float64],
    targets_train: NDArray[np.float64],
    learning_rate: float,
    num_epochs: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[float],
]:
    hidden_weights, hidden_bias, output_weights, output_bias = initialize_parameters()
    loss_history: list[float] = []

    for _ in range(num_epochs):
        total_epoch_loss = 0.0

        for sample_features, sample_target in zip(features_train, targets_train):
            (
                hidden_weights,
                hidden_bias,
                output_weights,
                output_bias,
                sample_loss,
            ) = train_one_sample(
                sample_features,
                float(sample_target),
                hidden_weights,
                hidden_bias,
                output_weights,
                output_bias,
                learning_rate,
            )

            total_epoch_loss += sample_loss

        average_epoch_loss = total_epoch_loss / len(features_train)
        loss_history.append(average_epoch_loss)

    return hidden_weights, hidden_bias, output_weights, output_bias, loss_history
