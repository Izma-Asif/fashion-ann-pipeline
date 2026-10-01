import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

model = keras.models.load_model("models/model.h5")

data = np.load("data/processed/fashion_mnist_processed.npz")
x_test, y_test = data["x_test"], data["y_test"]

loss, acc = model.evaluate(x_test, y_test)
y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.savefig("confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

print("Test accuracy:", acc)