import os
import json
import matplotlib.pyplot as plt

history_file = "training_history.json"
if not os.path.exists(history_file):
    print(f"Error: '{history_file}' not found.")
    exit(1)

with open(history_file, "r") as f:
    history = json.load(f)

epochs_range = range(1, len(history["accuracy"]) + 1)

# Combined Training Overview Graph
plt.figure(figsize=(14, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["accuracy"], "o-", label="Training Accuracy", color="#2b5c8f")
plt.plot(epochs_range, history["val_accuracy"], "s-", label="Validation Accuracy", color="#e27c38")
plt.title("Model Accuracy over Epochs", fontsize=13, pad=10)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["loss"], "o-", label="Training Loss", color="#2b5c8f")
plt.plot(epochs_range, history["val_loss"], "s-", label="Validation Loss", color="#d9534f")
plt.title("Model Loss over Epochs", fontsize=13, pad=10)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=300)
plt.savefig("loss_graph.png", dpi=300)
plt.close()

print("Graphs saved: accuracy_graph.png and loss_graph.png")