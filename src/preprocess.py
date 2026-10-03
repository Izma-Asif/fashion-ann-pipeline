import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

raw = np.load("data/raw/fashion_mnist_raw.npz")
x_train, y_train = raw["x_train"], raw["y_train"]
x_test, y_test = raw["x_test"], raw["y_test"]

x_train = x_train.astype("float32") / 127.5 - 1.0
x_test = x_test.astype("float32") / 127.5 - 1.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez(
    "data/processed/fashion_mnist_processed.npz",
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)