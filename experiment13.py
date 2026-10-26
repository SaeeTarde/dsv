
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

# --------------------------------------------------
# 1. Create the XOR truth table
# --------------------------------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

print("XOR Truth Table")
print("A B | XOR")
for (a, b), target in zip(X, y_xor):
    print(a, b, "|", target)


# --------------------------------------------------
# 2. Implement a single-layer perceptron
# --------------------------------------------------
class SimplePerceptron:
    """Single neuron using a step activation function."""

    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def step(self, z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        return self.step(X @ self.w + self.b)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.history = []

        for epoch in range(self.epochs):
            errors = 0

            for xi, target in zip(X, y):
                output = self.step(xi @ self.w + self.b)
                update = self.lr * (target - output)

                self.w = self.w + update * xi
                self.b = self.b + update
                errors += int(update != 0)

            self.history.append(errors)

            if errors == 0:
                break

        return self


# Function to verify predictions
def verify(model, X, y, name):
    predictions = model.predict(X)

    print("\n", name, "- Truth Table Verification")
    print("A B | Expected | Predicted | Result")

    for (a, b), expected, predicted in zip(X, y, predictions):
        result = "Correct" if expected == predicted else "WRONG"
        print(a, b, "|", expected, "|", predicted, "|", result)

    print("Accuracy:", (predictions == y).mean())


# Train the single neuron
single = SimplePerceptron(lr=0.1, epochs=100).fit(X, y_xor)

print("\nSingle Perceptron Results")
print("Errors in first 12 epochs:", single.history[:12])
print("Epochs run:", len(single.history))
print("Errors in last epoch:", single.history[-1])
print("Weights:", single.w.round(2))
print("Bias:", round(single.b, 2))

verify(single, X, y_xor, "XOR using a single neuron")

# Verify with scikit-learn Perceptron
p = Perceptron(max_iter=1000, tol=None, random_state=0)
p.fit(X, y_xor)

print("\nScikit-learn Perceptron accuracy:", p.score(X, y_xor))


# --------------------------------------------------
# 3. XOR using a hand-designed two-layer network
# XOR = AND(OR, NAND)
# --------------------------------------------------
def step(z):
    return (z >= 0).astype(int)


# Two hidden neurons: OR and NAND
W1 = np.array([
    [1, -1],
    [1, -1]
])

b1 = np.array([-0.5, 1.5])

# Output neuron: AND
W2 = np.array([1, 1])
b2 = -1.5

hidden = step(X @ W1 + b1)
output = step(hidden @ W2 + b2)

print("\nHand-Designed Network")
print("A B | OR NAND | XOR")

for (a, b), h, o in zip(X, hidden, output):
    print(a, b, "|", h[0], h[1], "|", o)

print("Matches XOR truth table:", (output == y_xor).all())


# --------------------------------------------------
# 4. Feed-forward neural network trained using
#    backpropagation from scratch (2-4-1 network)
# --------------------------------------------------
def sigmoid(z):
    # Clip values to help avoid numerical overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


rng = np.random.default_rng(0)

hidden_units = 4
lr = 0.5

# Initialise weights and biases
W1 = rng.normal(0, 1, (2, hidden_units))
b1 = np.zeros(hidden_units)

W2 = rng.normal(0, 1, (hidden_units, 1))
b2 = np.zeros(1)

t = y_xor.reshape(-1, 1)
loss_history = []

for epoch in range(10001):

    # Forward pass
    a1 = sigmoid(X @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)

    # Calculate mean squared error
    loss = np.mean((t - a2) ** 2)
    loss_history.append(loss)

    # Backpropagation
    d2 = (a2 - t) * a2 * (1 - a2)
    d1 = (d2 @ W2.T) * a1 * (1 - a1)

    # Update weights and biases
    W2 -= lr * (a1.T @ d2)
    b2 -= lr * d2.sum(axis=0)

    W1 -= lr * (X.T @ d1)
    b1 -= lr * d1.sum(axis=0)

    if epoch % 2000 == 0:
        print("Epoch", epoch, "Loss =", round(loss, 5))


# Verify the trained network
print("\nOutputs after backpropagation training:", a2.ravel().round(3))

pred = (a2 > 0.5).astype(int).ravel()

print("A B | Expected | Network Output | Result")
for (a, b), expected, value, predicted in zip(
        X, y_xor, a2.ravel(), pred):

    result = "Correct" if expected == predicted else "WRONG"
    print(a, b, "|", expected, "|", round(value, 3), "|", result)

print("Backpropagation accuracy:", (pred == y_xor).mean())


# --------------------------------------------------
# 5. XOR using scikit-learn MLPClassifier
# --------------------------------------------------
print("\nMLPClassifier Comparison")

for hidden_layer in [(2,), (4,)]:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer,
        activation="logistic",
        solver="lbfgs",
        max_iter=2000,
        random_state=1
    )

    mlp.fit(X, y_xor)

    print(
        "Hidden neurons:", hidden_layer[0],
        "| Iterations:", mlp.n_iter_,
        "| Loss:", round(mlp.loss_, 4),
        "| Accuracy:", mlp.score(X, y_xor)
    )


# Use four hidden neurons for final verification
best = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="logistic",
    solver="lbfgs",
    max_iter=2000,
    random_state=1
)

best.fit(X, y_xor)

print("\nProbability of output 1:",
      best.predict_proba(X)[:, 1].round(3))

verify(best, X, y_xor, "XOR using MLPClassifier")


# --------------------------------------------------
# 6. Plot training loss and decision regions
# --------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

# Loss curve
ax[0].plot(loss_history)
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Mean Squared Error")

# Create grid for decision-region plots
xx, yy = np.meshgrid(
    np.linspace(-0.5, 1.5, 200),
    np.linspace(-0.5, 1.5, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Compare single neuron and trained MLP
for a, model, title in [
    (ax[1], single, "Single Neuron (Fails)"),
    (ax[2], best, "MLP with Hidden Layer")
]:
    zz = model.predict(grid).reshape(xx.shape)

    a.contourf(
        xx, yy, zz,
        alpha=0.3,
        cmap="RdYlGn",
        levels=[-0.5, 0.5, 1.5]
    )

    a.scatter(
        X[y_xor == 0, 0],
        X[y_xor == 0, 1],
        s=140,
        color="tab:red",
        label="Output 0"
    )

    a.scatter(
        X[y_xor == 1, 0],
        X[y_xor == 1, 1],
        s=140,
        color="tab:green",
        label="Output 1"
    )

    a.set_title(title)
    a.set_xlabel("Input A")
    a.set_ylabel("Input B")
    a.legend(fontsize=8)

plt.tight_layout()
plt.show()
