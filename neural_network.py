import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.5):
        """
        Initialize the neural network with random weights and biases
        Architecture: 2 inputs -> 2 hidden neurons -> 1 output
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # W1: weights from input layer (2) to hidden layer (2) - shape (2, 2)
        self.W1 = np.random.uniform(-1, 1, (2, 2))
        
        # b1: biases for hidden layer - shape (1, 2)
        self.b1 = np.random.uniform(-1, 1, (1, 2))
        
        # W2: weights from hidden layer (2) to output layer (1) - shape (2, 1)
        self.W2 = np.random.uniform(-1, 1, (2, 1))
        
        # b2: bias for output layer - shape (1, 1)
        self.b2 = np.random.uniform(-1, 1, (1, 1))
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
        """
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation through the network
        X: input data (shape: batch_size, 2)
        Returns: output prediction and intermediate values
        """
        # Hidden layer: Z1 = X * W1 + b1
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply sigmoid activation to hidden layer
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer: Z2 = a1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Apply sigmoid activation to output layer
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss: L = (1/n) * Σ(y_true - y_pred)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        """
        Backpropagation: calculate gradients using chain rule
        """
        m = X.shape[0]  # number of samples
        
        # Calculate error at output layer
        # dL/da2 = -2 * (y_true - y_pred) / m
        output_error = -(y_true - y_pred) / m
        
        # Calculate gradient for output layer weights and bias
        # dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2
        # da2/dz2 = sigmoid_derivative(a2)
        # dz2/dW2 = a1
        delta_output = output_error * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)
        
        # Calculate error propagated to hidden layer
        # dL/da1 = dL/da2 * da2/dz2 * dz2/da1
        # dz2/da1 = W2
        hidden_error = np.dot(delta_output, self.W2.T)
        
        # Calculate gradient for hidden layer weights and bias
        # dL/dW1 = dL/da1 * da1/dz1 * dz1/dW1
        # da1/dz1 = sigmoid_derivative(a1)
        # dz1/dW1 = X
        delta_hidden = hidden_error * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """
        Update weights using gradient descent: W_new = W_old - η * dW
        """
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
    
    def train(self, X, y, epochs=10000):
        """
        Train the neural network using the complete dataset
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Calculate loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            
            # Update weights
            self.update_weights(dW1, db1, dW2, db2)
            
            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        return self.forward(X)

# XOR dataset
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1], 
              [0]])

# Create and train the network
print("Training Neural Network on XOR problem...")
print("=" * 50)

# Set random seed for reproducible results
np.random.seed(42)

# Create neural network
nn = NeuralNetwork(learning_rate=0.5)

# Print initial weights
print("\nInitial Weights:")
print(f"W1 (input to hidden):\n{nn.W1}")
print(f"b1 (hidden bias): {nn.b1}")
print(f"W2 (hidden to output):\n{nn.W2}")
print(f"b2 (output bias): {nn.b2}")

# Train the network
losses = nn.train(X, y, epochs=10000)

# Test the network
print("\nFinal Predictions:")
predictions = nn.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {predictions[i][0]:.4f}, Target: {y[i][0]}")

print(f"\nFinal Loss: {losses[-1]:.6f}")