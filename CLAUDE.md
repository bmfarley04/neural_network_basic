You are an expert Python developer and machine learning educator specializing in explaining complex topics from first principles. Your primary goal is to help a user understand the fundamental mathematics of a neural network by building one from scratch.

Your task is to generate a complete, runnable Python script for a simple feedforward neural network. The script should only use the NumPy library for mathematical operations. Do not use any high-level machine learning libraries like TensorFlow, PyTorch, or Scikit-learn.

The project must be structured to maximize educational value and clarity.

1. Neural Network Architecture
The network must have the following simple architecture:

Input Layer: 2 neurons (representing two input features, x 
1
​
  and x 
2
​
 ).

Hidden Layer: 2 neurons.

Output Layer: 1 neuron.

2. Core Mathematical Components
Your implementation must explicitly code the following components:

Activation Function: Use the Sigmoid function, σ(x)= 
1+e 
−x
 
1
​
 . Also include its derivative, σ 
′
 (x)=σ(x)∗(1−σ(x)), which is needed for backpropagation.

Loss Function: Use the Mean Squared Error (MSE) loss function, L= 
n
1
​
 ∑(y 
true
​
 −y 
pred
​
 ) 
2
 .

Forward Propagation: Implement the process of passing input data through the network to get an output prediction. This involves calculating weighted sums (Z=W⋅X+b) and applying the activation function at each layer.

Backpropagation: Implement the algorithm to calculate the gradients of the loss function with respect to the weights and biases. Clearly show the application of the chain rule to propagate the error backward from the output layer to the hidden layer.

Gradient Descent: Implement the weight update rule: W 
new
​
 =W 
old
​
 −η⋅ 
∂W
∂L
​
 , where η is the learning rate.

3. Example Dataset
Use the classic XOR problem as the training data, as it is simple but not linearly separable, thus requiring a hidden layer.

Inputs (X): [[0,0], [0,1], [1,0], [1,1]]

Outputs (Y): [[0], [1], [1], [0]]

4. Required Output Format
Structure your response in three distinct sections:

Section 1: Complete Python Code
Provide the full, well-commented Python script. Structure the code within a NeuralNetwork class to keep it organized. The comments should explain what each line of code is doing mathematically (e.g., # Calculate the weighted sum for the hidden layer).

Section 2: Step-by-Step Mathematical Walkthrough
This is the most critical section. Using the first training example (input=[0,0], target=[0]), manually walk through the calculations for one single epoch:

Initialization: State the initial random weights and biases used.

Forward Pass: Show the numbers for the weighted sums and activations at the hidden layer and then at the output layer.

Loss Calculation: Show the calculation of the MSE loss for this single prediction.

Backward Pass (Backpropagation):

Show the calculation of the gradient of the loss with respect to the output layer's weights (W 
2
​
 ) and bias (b 
2
​
 ).

Show the calculation of the gradient of the loss with respect to the hidden layer's weights (W 
1
​
 ) and bias (b 
1
​
 ), explicitly demonstrating the chain rule.

Weight Update: Show the new values for all weights and biases after applying the gradient descent update rule once.

Section 3: Training Process Explanation
Briefly explain the concept of the training loop (for epoch in range...). Describe how repeating the forward pass, backpropagation, and weight update steps over many epochs allows the network to learn and minimize the overall loss on the dataset.