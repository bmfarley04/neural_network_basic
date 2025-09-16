import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.W1 = np.random.uniform(-1, 1, (2, 2))
        self.b1 = np.random.uniform(-1, 1, (1, 2))
        self.W2 = np.random.uniform(-1, 1, (2, 1))
        self.b2 = np.random.uniform(-1, 1, (1, 1))
    
    def sigmoid(self, x):
        """σ(x) = 1 / (1 + e^(-x))"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """σ'(x) = σ(x) * (1 - σ(x))"""
        return x * (1 - x)
    
    def forward(self, X):
        """Z = W·X + b, a = σ(Z)"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """L = (1/n) * Σ(y_true - y_pred)²"""
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        """∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W (chain rule)"""
        m = X.shape[0]
        output_error = -(y_true - y_pred) / m
        delta_output = output_error * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)
        hidden_error = np.dot(delta_output, self.W2.T)
        delta_hidden = hidden_error * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """W_new = W_old - η * ∂L/∂W"""
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
    
    def train(self, X, y, epochs=10000):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            self.update_weights(dW1, db1, dW2, db2)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return losses
    
    def predict(self, X):
        return self.forward(X)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("Training Neural Network on XOR problem...")
print("=" * 50)

np.random.seed(42)
nn = NeuralNetwork(learning_rate=0.5)

print("\nInitial Weights:")
print(f"W1 (input to hidden):\n{nn.W1}")
print(f"b1 (hidden bias): {nn.b1}")
print(f"W2 (hidden to output):\n{nn.W2}")
print(f"b2 (output bias): {nn.b2}")

losses = nn.train(X, y, epochs=10000)

print("\nFinal Predictions:")
predictions = nn.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {predictions[i][0]:.4f}, Target: {y[i][0]}")

print(f"\nFinal Loss: {losses[-1]:.6f}")