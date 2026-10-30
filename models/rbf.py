import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def rbf_activation(
    sample: NDArray[np.float64],
    center: NDArray[np.float64],
    sigma: float,
) -> float:
    distance_squared = np.sum((sample - center) ** 2)
    return float(np.exp(-distance_squared / (2 * sigma**2)))


def initialize_centers(
    features_train: NDArray[np.float64],
    num_centers: int,
) -> NDArray[np.float64]:
    kmeans = KMeans(
        n_clusters=num_centers,
        random_state=42,
        n_init=10,
    )

    kmeans.fit(features_train)

    return kmeans.cluster_centers_


def calculate_sigma(
    centers: NDArray[np.float64],
) -> float:
    distances = []

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance = np.linalg.norm(centers[i] - centers[j])
            distances.append(distance)

    sigma = np.mean(distances)

    return float(sigma)


def transform_features(
    features: NDArray[np.float64],
    centers: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    transformed_features = np.zeros(
        (len(features), len(centers)),
        dtype=np.float64,
    )

    for i, sample in enumerate(features):
        for j, center in enumerate(centers):
            transformed_features[i, j] = rbf_activation(
                sample,
                center,
                sigma,
            )

    return transformed_features


def train_output_layer(
    transformed_features: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    features_with_bias = np.column_stack(
        (
            transformed_features,
            np.ones(len(transformed_features)),
        )
    )

    parameters, _, _, _ = np.linalg.lstsq(
        features_with_bias,
        targets,
        rcond=None,
    )

    weights = parameters[:-1]
    bias = float(parameters[-1])

    predictions = features_with_bias @ parameters
    loss = float(np.mean((predictions - targets) ** 2))

    return weights, bias, loss


def predict(
    features: NDArray[np.float64],
    centers: NDArray[np.float64],
    sigma: float,
    weights: NDArray[np.float64],
    bias: float,
) -> NDArray[np.float64]:
    transformed_features = transform_features(
        features,
        centers,
        sigma,
    )

    return transformed_features @ weights + bias


def train(
    features_train: NDArray[np.float64],
    targets_train: NDArray[np.float64],
    num_centers: int,
) -> tuple[
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    float,
    float,
]:
    centers = initialize_centers(features_train, num_centers)
    sigma = calculate_sigma(centers)

    transformed_features = transform_features(
        features_train,
        centers,
        sigma,
    )

    weights, bias, loss = train_output_layer(
        transformed_features,
        targets_train,
    )

    return centers, sigma, weights, bias, loss
