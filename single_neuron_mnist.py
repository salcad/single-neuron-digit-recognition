import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class SingleNeuron:
    def __init__(self, input_size: int, lr: float = 0.1, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=(input_size, 1))
        self.b = 0.0
        self.lr = lr

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x @ self.w + self.b)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps)
        )

    def train_step(self, x: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.forward(x)
        m = x.shape[0]
        dz = y_pred - y_true
        dw = (x.T @ dz) / m
        db = float(np.mean(dz))
        self.w -= self.lr * dw
        self.b -= self.lr * db
        return float(self.compute_loss(y_true, y_pred))

    def train(self, x: np.ndarray, y_true: np.ndarray, epochs: int = 100, verbose: bool = True):
        losses = []
        for epoch in range(epochs):
            loss = self.train_step(x, y_true)
            losses.append(loss)
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.6f}")
        return losses

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.int32)


def load_binary_mnist(test_size: float, random_state: int):
    x, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )
    mask = (y == "0") | (y == "1")
    x = x[mask].astype(np.float32) / 255.0
    y = np.where(y[mask] == "0", 0.0, 1.0).astype(np.float32).reshape(-1, 1)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y.ravel(),
    )


def accuracy(model: SingleNeuron, x: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y_true))


def save_loss_plot(losses, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color="tab:blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Single Neuron Learning Curve on MNIST 0 vs 1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_model(model: SingleNeuron, weights_path: Path, bias_path: Path):
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    bias_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(weights_path, model.w)
    np.save(bias_path, np.array(model.b, dtype=np.float32))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single-neuron classifier for MNIST digits 0 and 1."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("artifacts/loss_curve.png"),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the loss curve plot.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("artifacts/neuron_weights.npy"),
    )
    parser.add_argument(
        "--bias-path",
        type=Path,
        default=Path("artifacts/neuron_bias.npy"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, x_test, y_train, y_test = load_binary_mnist(
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = SingleNeuron(input_size=x_train.shape[1], lr=args.lr, seed=args.seed)
    losses = model.train(x_train, y_train, epochs=args.epochs, verbose=True)

    train_acc = accuracy(model, x_train, y_train)
    test_acc = accuracy(model, x_test, y_test)

    print()
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    save_model(model, args.weights_path, args.bias_path)
    print(f"Weights saved to: {args.weights_path}")
    print(f"Bias saved to: {args.bias_path}")

    if not args.no_plot:
        save_loss_plot(losses, args.plot_path)
        print(f"Loss curve saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
