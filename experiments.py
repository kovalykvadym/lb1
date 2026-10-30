import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import calculate_metrics
from models import perceptron, mlp, rbf
from prepare_data import get_prepared_data

LEARNING_RATES = [0.001, 0.01, 0.1]
EPOCHS = 20
NUM_CENTERS = 32


def save_results(
    results: list[dict[str, str | float]],
    filename: str,
) -> None:
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


def save_model_parameters(
    parameters: dict[str, np.ndarray | float],
    filename: str,
) -> None:
    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    np.savez(output_path, **parameters)

    print(f"Model parameters saved to: {output_path}.npz")


def plot_training_time(
    results: list[dict[str, float]],
    model_name: str,
) -> None:
    learning_rates = [result["learning_rate"] for result in results]

    training_times = [result["training_time"] for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(
        learning_rates,
        training_times,
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Time (seconds)")
    plt.title(f"{model_name}: Training Time vs learning rate")
    plt.grid(True)
    plt.tight_layout()

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name.lower()}_training_time.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {output_path}")


def run_perceptron_experiments(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    features_test: np.ndarray,
    targets_test: np.ndarray,
) -> list[dict[str, float]]:
    results = []

    for learning_rate in LEARNING_RATES:
        start_time = time.perf_counter()

        weights, bias, loss_history = perceptron.train(
            features_train,
            targets_train,
            learning_rate,
            EPOCHS,
        )

        training_time = time.perf_counter() - start_time

        save_model_parameters(
            {
                "weights": weights,
                "bias": bias,
                "learning_rate": learning_rate,
                "input_size": features_train.shape[1],
                "output_size": 1,
            },
            f"perceptron_lr_{learning_rate}",
        )

        predictions = perceptron.predict(
            features_test,
            weights,
            bias,
        )

        metrics = calculate_metrics(targets_test, predictions)

        results.append(
            {
                "learning_rate": learning_rate,
                "training_time": training_time,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
                "accuracy": metrics["accuracy"],
                "classification_error": metrics["classification_error"],
            }
        )

    return results


def run_mlp_experiments(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    features_test: np.ndarray,
    targets_test: np.ndarray,
) -> list[dict[str, float]]:
    results = []

    for learning_rate in LEARNING_RATES:
        start_time = time.perf_counter()

        (
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            loss_history,
        ) = mlp.train(
            features_train,
            targets_train,
            learning_rate,
            EPOCHS,
        )

        training_time = time.perf_counter() - start_time

        save_model_parameters(
            {
                "hidden_weights": hidden_weights,
                "hidden_bias": hidden_bias,
                "output_weights": output_weights,
                "output_bias": output_bias,
                "learning_rate": learning_rate,
                "input_size": features_train.shape[1],
                "hidden_size": hidden_weights.shape[1],
                "output_size": output_weights.shape[1],
            },
            f"mlp_lr_{learning_rate}",
        )

        predictions = mlp.predict(
            features_test,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        )

        metrics = calculate_metrics(targets_test, predictions)

        results.append(
            {
                "learning_rate": learning_rate,
                "training_time": training_time,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
                "accuracy": metrics["accuracy"],
                "classification_error": metrics["classification_error"],
            }
        )

    return results


def run_rbf_experiment(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    features_test: np.ndarray,
    targets_test: np.ndarray,
) -> dict[str, float]:
    start_time = time.perf_counter()

    centers, sigma, weights, bias, loss = rbf.train(
        features_train,
        targets_train,
        NUM_CENTERS,
    )

    training_time = time.perf_counter() - start_time

    save_model_parameters(
        {
            "centers": centers,
            "sigma": sigma,
            "weights": weights,
            "bias": bias,
            "num_centers": NUM_CENTERS,
            "input_size": features_train.shape[1],
            "output_size": 1,
        },
        "rbf",
    )

    predictions = rbf.predict(
        features_test,
        centers,
        sigma,
        weights,
        bias,
    )

    metrics = calculate_metrics(targets_test, predictions)

    return {
        "num_centers": float(NUM_CENTERS),
        "training_time": training_time,
        "final_loss": loss,
        "accuracy": metrics["accuracy"],
        "classification_error": metrics["classification_error"],
    }


def main() -> None:
    (
        features_train,
        features_test,
        targets_train,
        targets_test,
        scaler,
    ) = get_prepared_data()

    save_model_parameters(
        {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        },
        "scaler",
    )

    perceptron_results = run_perceptron_experiments(
        features_train,
        targets_train,
        features_test,
        targets_test,
    )

    mlp_results = run_mlp_experiments(
        features_train,
        targets_train,
        features_test,
        targets_test,
    )

    rbf_results = run_rbf_experiment(
        features_train,
        targets_train,
        features_test,
        targets_test,
    )

    save_results(perceptron_results, "perceptron_results.csv")
    save_results(mlp_results, "mlp_results.csv")
    save_results([rbf_results], "rbf_results.csv")

    comparison_results = [
        {"model": "Perceptron", **result} for result in perceptron_results
    ]

    comparison_results.extend({"model": "MLP", **result} for result in mlp_results)

    comparison_results.append({"model": "RBF", **rbf_results})

    save_results(comparison_results, "comparison_results.csv")

    plot_training_time(perceptron_results, "Perceptron")
    plot_training_time(mlp_results, "MLP")

    print("Perceptron results:")
    for result in perceptron_results:
        print(result)

    print("\nMLP results:")
    for result in mlp_results:
        print(result)

    print("\nRBF results:")
    print(rbf_results)


if __name__ == "__main__":
    main()
