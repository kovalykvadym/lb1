from pathlib import Path

import numpy as np

models_dir = Path("results/models")

for model_path in sorted(models_dir.glob("*.npz")):
    print(f"\n{model_path.name}")

    with np.load(model_path) as parameters:
        for name in parameters.files:
            value = parameters[name]
            print(f"\t{name}: {value.shape if value.ndim > 0 else value.item()}")
