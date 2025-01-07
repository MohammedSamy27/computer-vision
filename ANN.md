---
share_link: https://share.note.sx/i5jl9jqd#81TEY7RD9XroMXInRHGmqp/d87bjOF5o0MniRJueT1E
share_updated: 2025-01-01T06:00:03+02:00
---
Artificial neural networks are brain-inspired systems used in machine learning to mimic human learning. They consist of input, output, and often hidden layers to process data and recognize complex patterns. Advances like backpropagation and deep learning have significantly improved their effectiveness in tasks like pattern recognition and feature extraction.

## **Basic Structure of ANNs**
- **Concept:** ANNs are inspired by the human brain, imitating its functioning using nodes and connections as neurons and axons.
- **Human Brain Comparison:** The brain has 86 billion neurons, connected by axons, which transmit electric impulses based on inputs from sensory organs.

	![[Pasted image 20241225114618.png]]


> [!NOTE] **Structure of ANNs**
> 1. **Input Layer:**
> 	- The input layer receives the raw data features (inputs) and feeds them into the network.
> 	- Each node (neuron) in this layer corresponds to a specific feature or attribute in the input data.
> 	- The number of input nodes is equal to the number of features in the dataset.
> 2. **Hidden Layers:**
> 	- These layers lie between the input and output layers and are responsible for most of the computation and learning.
> 	- A neural network can have one or more hidden layers.
> 	- Each neuron in the hidden layer processes inputs from the previous layer and produces an output, which is passed to the next layer.
> 	- The complexity of the model increases with the number of hidden layers, enabling 
> 	- the network to learn more complex patterns (especially in deep neural networks).
> 3. **Output Layer:**
> 	- The output layer produces the final output of the network, which could be a class label (for classification) or a continuous value (for regression).
> 	- The output layer produces the final output of the network, which could be a class label (for classification) or a continuous value (for regression).
> 		1. For binary classification, there is typically one output node.
> 		2. For multi-class classification, the number of output nodes matches the number of classes.
> 		3. For regression, there is usually one output node.
> 4. **Neurons (Nodes):**
> 	-  Each neuron in the network performs a simple computation, typically the sum of weighted inputs followed by an activation function.
> 	- The neurons are connected by links (synapses) with weights that represent the importance of each input.
> 5.  **Weights and Biases:**
> 	- **Weights:** Each connection between neurons has an associated weight that determines the strength of the connection.
> 	- **Biases:** Each neuron has a bias that helps adjust the output, providing additional flexibility in the model.
> 6. **Activation Function:**
> 	- Each neuron applies an activation function to its output before passing it to the next layer.
> 	- Common activation functions include Sigmoid, ReLU, Tanh, and Softmax, which introduce non-linearity into the model, allowing it to learn complex patterns.
> 7. **Connections Between Layers:**
> 	- The neurons in each layer are connected to the neurons in the next layer through weighted connections.
> 	- The output from one layer becomes the input to the next layer.
> 8. **Feedforward Process:** In the feedforward process, data moves from the input layer through the hidden layers to the output layer, where predictions or classifications are made.
> 9. **Backpropagation Process:** After the network generates an output, backpropagation is used to adjust the weights based on the error, improving the model's accuracy during training.
> ![[Pasted image 20241225114654.png]]


- **Learning Mechanism:**
    - Each connection has an associated weight.
    - Learning happens by adjusting these weights to improve performance.
- **Functionality:**
    - Inputs are processed through the network, mimicking the way neurons interact and transmit signals.

		
## **Types of Artificial Neural Networks**
There are two Artificial Neural Network topologies  **Feedforward** and **Feedback.**

### **1. Feedforward ANN**
In this ANN, the information flow is unidirectional. A unit sends information to another unit from which it does not receive any information. There are no feedback loops. They are used in pattern generation/recognition/classification. They have fixed inputs and outputs.

![[Pasted image 20241225115153.png]]

### **2. Feedback ANN**
Feedback loops are allowed. They are used in content-addressable memories.
![[Pasted image 20241225115436.png]]

## **Activation Function**
- **Definition:** An activation function (also called a transfer function) determines the output of a node in an ANN.
- **Importance:**
    1. Allows the ANN to model non-linear complex relationships.
	2. Converts input signals into output signals for use in the next layer.
- **Role in ANN:**
    - Computes the sum of products of inputs $(X)$ and their weights $(W)$.
    - Applies the activation function ($f(x)$) to this sum to produce the output of a layer.
    - The output becomes the input for the subsequent layer.
- **Purpose:**Introduces non-linear properties, enabling the network to learn and generalize better.

### **The Activation Functions can be based on 2 types**
	1. Linear Activation Function
	2. Non-linear Activation Functions

#### **1. Linear Activation Function**
**Equation:** $$f(x) = x$$
**Range:** $(-∞, ∞)$

![[Pasted image 20241225121003.png]]

#### 2. **Non-linear Activation Function**
- Nonlinear activation functions are commonly used because they help the model generalize better and differentiate between outputs.
- They allow the network to adapt to various types of data.
- They make the graph more complex, enabling the network to model non-linear relationships.

![[Pasted image 20241225122746.png]]

- **Key Terminologies:**
    - **Derivative (Slope):** The change in the output (y-axis) relative to the change in the input (x-axis).
    - **Monotonic Function:** A function that is either entirely non-increasing or non-decreasing.

##### **Types of Nonlinear Activation Functions:**  
They are categorized based on their range or curve shape.

> [!NOTE] **1. Sigmoid Activation Function**
> - **Range:** The output of the sigmoid function lies between 0 and 1, making it ideal for predicting probabilities.
> - **Differentiability:** The sigmoid function is differentiable, allowing the slope to be calculated at any point.
> - **Monotonicity:** The function is monotonic, but its derivative is not monotonic.
> - **Training Limitation:** The sigmoid function can cause a neural network to get stuck during training (due to issues like vanishing gradients).
> ![[Pasted image 20241225123259.png]]
> - **Softmax Function:**
> 	- **Use Case:** The softmax function is a more generalized form of the sigmoid, commonly used for multiclass classification problems.
> 	- **Functionality:** It outputs probabilities for each class in a multiclass scenario, ensuring the sum of probabilities across all classes is 1.


> [!NOTE] **2. Tanh Activation Function**
> - **Range:** The output of the tanh function lies between -1 and 1, unlike the sigmoid function which lies between 0 and 1.
> - **Shape:** The tanh function is sigmoidal (S-shaped) like the sigmoid function.
> ![[Pasted image 20241225124226.png]]
> -  **Advantages:**
> 	- Strongly negative inputs are mapped to negative values, and inputs near zero are mapped close to zero in the tanh graph.
> 	- This helps in better handling of both positive and negative values, improving performance.
> - **Differentiability:** The tanh function is differentiable, allowing the slope to be calculated at any point.
> - **Monotonicity:** The function is monotonic, but its derivative is not.
> - **Use Case:** Primarily used for binary classification tasks (classifying between two classes).
> - **Common Usage:** Both tanh and logistic sigmoid functions are used in feed-forward neural networks.


> [!NOTE] **3. ReLU (Rectified Linear Unit) Activation Function**
> - ReLU is a half-rectified function
> 	- For values less than 0, the output is 0.
> 	- For values greater than or equal to 0, the output is equal to the input ($f(z)=z$).
> 	![[Pasted image 20241225124842.png]]
> - **Range:** The output range of ReLU is $[0, ∞]$.
> - **Monotonicity:** Both the function and its derivative are monotonic.
> - **Limitation:**
> 	- Negative values are mapped to zero, which can hinder the model's ability to learn from negative inputs.
> 	- This issue is known as the "dying ReLU problem," where negative inputs are lost, reducing the model's capacity to train effectively.


## **Training**
1. **Initialize Weights:** Randomly initialize weights close to 0 (but not exactly 0).
2. **Input Data:** Feed the first observation of your dataset into the input layer, with each feature mapped to one input node.
3. **Forward Propagation:**
    - Propagate the inputs from left to right through the network, where each neuron's activation is weighted.
    - Continue propagating until you get the predicted output $y$.
4. **Calculate Error:** Compare the predicted output to the actual result and measure the error.
5. **Backpropagation:**
    - Propagate the error back through the network (from right to left).
    - Update the weights based on their contribution to the error, using the learning rate to determine how much to adjust the weights.



> [!quote] Next
> [[CNN]]