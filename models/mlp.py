import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from prepare_data import get_prepared_data


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
    sample_features: NDArray[np.float64],
    hidden_weights: NDArray[np.float64],
    hidden_bias: NDArray[np.float64],
    output_weights: NDArray[np.float64],
    output_bias: NDArray[np.float64],
) -> float:
    hidden_input = sample_features @ hidden_weights + hidden_bias
    hidden_activation = activation(hidden_input)

    output_input = hidden_activation @ output_weights + output_bias
    prediction = activation(output_input)

    return float(prediction[0])


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


def main() -> None:
    features_train, features_test, targets_train, targets_test, _ = get_prepared_data()

    learning_rate = 0.01
    num_epochs = 20

    start_time = time.perf_counter()

    hidden_weights, hidden_bias, output_weights, output_bias, loss_history = train(
        features_train,
        targets_train,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    training_time = time.perf_counter() - start_time

    predictions = np.array(
        [
            predict(
                sample_features,
                hidden_weights,
                hidden_bias,
                output_weights,
                output_bias,
            )
            for sample_features in features_test
        ]
    )

    predicted_labels = np.where(predictions >= 0, 1.0, -1.0)

    num_correct = int(np.sum(predicted_labels == targets_test))
    accuracy_percent = num_correct / len(targets_test) * 100

    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Initial average loss: {loss_history[0]:.6f}")
    print(f"Final average loss: {loss_history[-1]:.6f}")
    print(f"Correct predictions: {num_correct} / {len(targets_test)}")
    print(f"Classification accuracy: {accuracy_percent:.2f}%")

    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title(f"MLP training loss (learning rate = {learning_rate})")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
