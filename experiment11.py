
# Experiment 11: AND/NAND Gate using Neural Networks

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

# Step 1: Create truth tables
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])
y_nand = np.array([1, 1, 1, 0])

print("A B | AND NAND")
for (a, b), and_out, nand_out in zip(X, y_and, y_nand):
    print(a, b, "| ", and_out, "   ", nand_out)


# Step 2: Define a simple perceptron
class SimplePerceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    # Step activation function
    def step(self, z):
        return np.where(z >= 0, 1, 0)

    # Predict outputs
    def predict(self, X):
        return self.step(X @ self.w + self.b)

    # Train the perceptron
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            errors = 0

            for xi, target in zip(X, y):
                output = self.step(xi @ self.w + self.b)
                update = self.lr * (target - output)

                self.w = self.w + update * xi
                self.b = self.b + update

                if update != 0:
                    errors += 1

            print(
                f"Epoch {epoch}: errors={errors}, "
                f"weights={self.w.round(2)}, "
                f"bias={round(self.b, 2)}"
            )

            if errors == 0:
                print("Converged after", epoch, "epochs")
                break

        return self


# Step 3: Verify predictions
def verify(model, X, y, name):
    predictions = model.predict(X)

    print(f"\n{name} Gate Verification")
    print("A B | Expected | Predicted | Result")

    for (a, b), expected, predicted in zip(X, y, predictions):
        result = "Correct" if expected == predicted else "Wrong"
        print(
            f"{a} {b} |    {expected}     |     "
            f"{predicted}     | {result}"
        )

    print("Accuracy:", np.mean(predictions == y))


# Step 4: Train the AND gate
print("\n--- Training AND Gate ---")
and_model = SimplePerceptron(lr=0.1, epochs=50)
and_model.fit(X, y_and)
verify(and_model, X, y_and, "AND")


# Step 5: Train the NAND gate
print("\n--- Training NAND Gate ---")
nand_model = SimplePerceptron(lr=0.1, epochs=50)
nand_model.fit(X, y_nand)
verify(nand_model, X, y_nand, "NAND")


# Step 6: Train a feed-forward neural network
print("\n--- MLPClassifier Results ---")

for name, target in [("AND", y_and), ("NAND", y_nand)]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="logistic",
        solver="lbfgs",
        max_iter=1000,
        random_state=1
    )

    mlp.fit(X, target)
    predictions = mlp.predict(X)

    print(f"\n{name} Gate")
    print("Predictions:", predictions)
    print("Expected:   ", target)
    print("Accuracy:", np.mean(predictions == target))
    print("Iterations:", mlp.n_iter_)
    print("Output probabilities:", mlp.predict_proba(X)[:, 1].round(3))


# Step 7: Compare using scikit-learn Perceptron
print("\n--- Scikit-learn Perceptron ---")

for name, target in [("AND", y_and), ("NAND", y_nand)]:
    model = Perceptron(
        max_iter=1000,
        tol=None,
        random_state=0
    )
    model.fit(X, target)

    print(
        name,
        "Accuracy:", model.score(X, target),
        "Weights:", model.coef_[0],
        "Bias:", model.intercept_[0]
    )


# Step 8: Plot decision boundaries for our perceptrons
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, name, model, target in [
    (axes[0], "AND Gate", and_model, y_and),
    (axes[1], "NAND Gate", nand_model, y_nand)
]:
    ax.scatter(
        X[target == 0, 0], X[target == 0, 1],
        s=120, label="Output 0"
    )
    ax.scatter(
        X[target == 1, 0], X[target == 1, 1],
        s=120, marker="s", label="Output 1"
    )

    xs = np.linspace(-0.5, 1.5, 100)

    # Decision boundary: w1*A + w2*B + b = 0
    if model.w[1] != 0:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        ax.plot(xs, ys, "k--", label="Decision boundary")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.set_title(name)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nExperiment 11 completed successfully!")
