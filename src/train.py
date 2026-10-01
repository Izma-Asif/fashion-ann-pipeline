import os
import numpy as np
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

data = np.load("data/processed/fashion_mnist_processed.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[keras.callbacks.CSVLogger("models/history.csv")],
)

model.save("models/model.h5")
print("Saved models/model.h5 and models/history.csv")