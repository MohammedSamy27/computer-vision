---
share_link: https://share.note.sx/70p8p0zj#yCzt95rMLAsj/7uNRxhMIrRCBzJpfeNyp1xCaInC+84
share_updated: 2025-01-01T06:00:13+02:00
---
Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed for processing structured data like images, videos, and spatially organized information. They are widely used in computer vision tasks, such as image classification, object detection, and segmentation.

Convolutional Neural Network (CNN) is the extended version of artificial neural networks (ANN) which is predominantly used to extract the feature from the grid-like matrix dataset. For example visual datasets like images or videos where data patterns play an extensive role.

## **Structure of CNN**  
CNNs consist of several layers designed to process and interpret input data hierarchically. The primary layers are:
- **Input layer:** Accepts the raw input data, such as an image, which is then processed by subsequent layers.
- **Convolutional layer:** Extracts features from the input by applying filters.
- **Pooling layer:** Reduces the spatial dimensions of the data to decrease computational requirements and improve generalization.
- **Dense layers**
- **Fully connected layer:** Integrates features learned by previous layers to make predictions.
	![[Simple CNN architecture.png]]

## **Functions of Each Layer:**

- ### **1. Convolutional layer**
	- Convolutional layers are the **core building blocks** of Convolutional Neural Networks (CNNs).
	- They perform a critical operation called **convolution**, enabling the network to learn visual patterns.
	- The process involves applying specialized filters called **kernels** to the input image.
	- These filters move (or traverse) across the image, capturing features like edges, textures, and shapes.
	
	- ##### **Kernels** 
		- **Definition:** Kernels are small **matrices of numbers** that are used in convolutional layers of CNNs. These matrices slide across the image, performing **element-wise multiplication** with the corresponding pixel values. The result of these multiplications is summed and recorded in the output matrix, forming a **feature map**.
		- **Function of Kernels:**
			Kernels are responsible for extracting features from images, such as:
			    1. **Edges:** Detecting boundaries within an image.
			    2. **Textures:** Identifying patterns like smooth or rough areas.
			    3. **Shapes:** Capturing geometric structures like corners or curves.
					![[kernalsOperation.gif]]
		- **How Kernels Work:**
			- Visualize an image as a grid of pixel values (e.g., grayscale or RGB).
			- A kernel (e.g., a 3×3 matrix) slides over the image:
			    - At each position, the kernel values are multiplied with the corresponding pixel values it covers.
			    - The resulting values are summed and stored in the output matrix.
			- This process is repeated across the entire image, generating a transformed representation in the feature map.
		
	- **Convolution Operation**  
		The convolution operation involves multiplying **the kernel value**s by the **original pixel values** of the image and then **summing up the results**.
		
		This is a basic example with a 2 _×_ 2 kernel:
				![[cnn operation.png]]
			
		- Step-by-step computation
			1. Start from the **top-left corner** of the input matrix:
				$$(0 × 0) + (1 × 1) + (3 × 2) + (4 × 3) = 19$$
			2. Move **one pixel to the right**:
				$$(1 × 0) + (2 × 1) + (4 × 2) + (5 × 3) = 25$$
			3. Move to the **next row (down)** and start from the left:
				$$(3 × 0) + (4 × 1) + (6 × 2) + (7 × 3) = 37$$
			4. Move **one pixel to the right** again:
				$$(4 × 0) + (5 × 1) + (7 × 2) + (8 × 3) = 43$$
		- **Key Points:**
			- The size of the feature map depends on:
			    1. The size of the input matrix.
			    2. The size of the kernel.
			    3. The stride (how far the kernel moves each step).
			- The feature map highlights patterns like edges or textures, which are used for further processing in the network.
		This operation allows CNNs to learn spatial hierarchies of features from input data.
		
	- ##### **Channels**
		- **Color Channels in Images**  
			Digital images are often composed of three color channels: **Red, Green, and Blue (RGB)**. These are represented as three separate matrices, each corresponding to one color channel.
			For RGB images, **kernels typically process each channel independently** to capture features specific to that channel. For instance:
			- Some patterns may stand out more in the red channel.
			- Other details, like textures, might be more pronounced in green or blue.
				![[Convolution operation in Red, Green, and Blue channels.gif]]
				
		- **Depth of a Convolutional Layer**  
			The **depth** of a convolutional layer refers to the number of kernels (or filters) it contains.
			- Each kernel extracts a specific feature, creating a **feature map** as its output.
			- The **output of the layer** consists of multiple feature maps, collectively forming a three-dimensional representation of the processed image.
			
		- **Customizing Channels for Feature Detection**  
			While RGB images have three natural color channels, convolutional layers allow you to add as many channels as needed.
			
			- For instance, consider a grayscale image of a cat:
			    - One channel could be specialized in detecting the **ears**.
			    - Another channel could focus on the **mouth**.
			
			Each channel operates with a unique kernel that is fine-tuned during training to detect specific features relevant to the task.
				![[CNNrepresentation.png]]
				
		- **Clarifying Channels in Convolution**  
			It’s important to distinguish between:
			- **Color channels** in the input image (e.g., RGB).
			- **Feature channels** in the convolutional layer output, created by applying multiple kernels.
			
			The concept of channels in convolutional layers represents **abstract features**, such as edges, textures, or shapes, rather than colors.
			
		Channels in convolutional layers are a powerful way to detect diverse and meaningful features in an image. By assigning specific kernels to each channel, CNNs can learn to recognize complex patterns, enhancing their effectiveness in tasks like object recognition and image analysis.
	
	- ##### **Stride**
		We’ve discussed how a kernel moves through the pixels of an image during convolution, but we haven’t yet explored the different ways in which it can move.
		**Stride** refers to the number of pixels by which the kernel moves as it slides across the input image.
		
		- **Effect of Stride on Convolution**
			- In the previous example, we used a **stride of 1**, meaning the kernel moved one pixel at a time.
			
			- But the stride can be adjusted to move faster or slower across the image. Let’s see the difference:
				- **Stride = 1:** The kernel moves one pixel at a time. The output feature map is larger, as the kernel takes more steps.
					![[stride1.gif]]
					
				- **Stride = 2:** The kernel moves two pixels at a time. The output feature map becomes smaller because the kernel covers more pixels at once.
					![[stride2.gif]]
					
		- **Output Dimensions and Stride**
			- A **larger stride** will make the output feature map smaller, as the kernel covers more area in fewer steps.
			- A **smaller stride** results in a larger feature map, as the kernel moves more slowly and captures more local information.
			
		- **Why Change the Stride?**  
			Adjusting the stride has practical benefits:
			
			- **Larger Stride (e.g., stride = 2):**
			    - **Captures more global features** of the image.
			    - Helps **reduce overfitting** by reducing the spatial dimensions of the feature map.
			    - **Improves computational efficiency** by reducing the size of the output, speeding up training.
			- **Smaller Stride (e.g., stride = 1):**
			    - **Captures finer details** of the image, useful for detecting intricate patterns.
			    - Results in a larger output, which can help preserve detailed information but increases computation.
			    
		The stride is a key parameter that balances **global feature extraction** and **local detail capture**. Adjusting the stride allows you to control the output size, computational efficiency, and model’s ability to generalize, making it an important factor in convolutional layer design.
		
	- **Padding**
		Padding refers to the **addition of extra pixels** around the edge of the input image. This is done to ensure that the kernel has enough space to process the image’s edge pixels effectively.
		
		- **Why Padding is Important**  
			When applying convolution, pixels at the edges of the image are typically **traversed fewer times** than those in the center. Without padding, the kernel might miss important features at the borders of the image. Padding addresses this issue by giving the kernel more access to edge pixels.
			
		- **Example of Padding in Action**  
			Let’s compare two scenarios:
			
			- **Padding = 0:** No padding is applied. The kernel only processes the central pixels, leaving edge pixels with fewer passes. This means the model may miss key information at the borders.
				![[padding0.gif]]
			
			- **Padding = 1:** Padding is added around the input image. The kernel now has more opportunities to interact with edge pixels, capturing more detailed information about the borders.
				![[padding1.gif]]
				
		- **When to Apply Padding**  
			Padding is particularly useful when the **edges of the image contain important features** that you want the model to capture. For example, in object detection tasks, crucial features might be located near the edges, and padding helps the kernel focus on them.
			
		- **Effect on Output Feature Map**  
			Padding increases the size of the **output feature map** because it gives the kernel more "room" to operate. When you add padding while keeping the kernel size and stride constant, the convolution operation has more space to slide, resulting in a larger output.
			
			- **Formula for Output Size:**  
			    The output size of a convolutional layer can be calculated as:
				    $${Output Size} = (\frac{{{InputSize}−{KernelSize}}+{2×Padding}}{Stride})+1$$
			    
				Where:
				
				- **2 × Padding** accounts for the added pixels on both sides of the input image.
				- **+ 1** accounts for the initial position where the filter starts, at the beginning of the padded input.
				
		- **Asymmetry and Custom Padding**  
			Padding doesn’t always have to be symmetric. You can apply different padding sizes to different sides of the input image, or even design custom padding strategies based on the task at hand. This flexibility is useful when you want to control how much of the image's borders are considered in the convolution.
			
		Padding is a critical aspect of convolutional operations, especially for capturing features near the edges of the image. By adjusting padding, you can control the spatial dimensions of the output, enhance feature extraction, and improve model performance.
- ### 2. **Pooling layer:**
	- **Clarifying the Purpose of Convolutional Layers**  
		Before diving into the details of how convolutional layers work, it's important to clarify a common misconception:
		
		- **Convolutional layers are not primarily for dimensionality reduction.**
		- Their **main purpose is feature extraction**.
		
		Even though convolutional layers might reduce the spatial dimensions of the feature map slightly, the overall goal is to extract important features from the input image. As a result, while the feature map dimensions might get smaller, the number of channels typically increases, which means that more features are being captured.
		
		In essence, **convolutional layers enhance feature representation**, not reduce the data’s dimensions.
		
	- **Pooling Layers and Dimensionality Reduction**  
		In contrast, the purpose of **pooling layers** is indeed **dimensionality reduction**. These layers reduce the spatial size of the feature map while retaining important features such as edges, textures, and colors.
			![[Convolutional neural network representation.png]]
			
	- **How Pooling Layers Work**  
		Imagine you have a large image, and you want to shrink it but retain essential features like edges and colors. Pooling layers achieve this by performing operations like **Max Pooling** or **Average Pooling**.
		1. **Max Pooling** takes the maximum value in a given window.
		2. **Average Pooling** computes the average of the values in that window.
		
		These operations are applied independently to each depth slice of the input, effectively resizing the feature map. For example, a **4x4** feature map could be reduced to a **2x2** map, while preserving critical information.
		
		![[Max and Avg Pooling Layers.gif]]
		
	- **Difference Between Pooling and Convolution**  
		Unlike convolution, where a kernel (filter) is applied to the input to extract features, pooling layers do not use kernels. Instead, they simplify the information through mathematical operations (either max or average). Pooling focuses on reducing the spatial dimensions without modifying the number of channels.
		
	- **Pooling and Channels**  
		It’s important to note that **pooling layers do not reduce the number of channels**. Pooling is performed independently on each channel of the input data. For example, if your input has 3 channels (such as an RGB image), pooling will reduce the spatial dimensions of each channel separately but will **not decrease the number of channels**.
		
		In other words, the number of channels remains the same throughout the pooling layers. The reduction only affects the spatial dimensions of each channel.
		
		![[Layers inside a CNN .png]]
		
	- **Why and How Do We Combine These Channels?**  
		After convolutional and pooling layers have extracted relevant features, the output is a high-dimensional feature map, typically with many channels. To feed this information into **fully connected layers** (which expect a 1D input), we need to **flatten** this multi-dimensional output into a 1D vector.
		
		This is where **flattening layers** come into play, transforming the high-dimensional feature map into a format suitable for fully connected layers.
		
- ### **3. Flattening Layers**
	Flattening is a process where the entire **feature map** (which may consist of multiple channels) is **reorganized into a single, long vector**. Imagine a grid of data, like the pixels in a feature map. Flattening takes this grid and “flattens” it, so all the data points are arranged in one continuous line.
	
	This process **does not change the information**; it merely alters the structure of the data to fit the next layer.
	
	![[Flattening concept.png]]
	
	- **Why Do We Need Flattening Layers?**  
		Flattening is an essential step in a CNN for several reasons:
		- **Integration of Features:**  
		    Flattening allows the network to integrate all the spatially distributed features extracted by the convolutional and pooling layers. For tasks like image classification, the network needs to combine these features into a cohesive decision-making process, and flattening enables this by converting multi-dimensional data into a 1D vector.
		    
		- **Compatibility with Dense Layers:**  
		    **Dense layers** (or fully connected layers) are designed to process 1D data. Convolutional layers produce multi-dimensional tensors (i.e., they output feature maps with height, width, and depth). To connect these feature maps to the fully connected layers, the multi-dimensional data must be **flattened** into a 1D vector that can be fed into the dense layer.
		    
	 - **Why Do We Need Dense Layers in CNNs?**
			**1. Role of Dense Layers**  
				While convolutional layers are effective at detecting low-level features (like edges, textures, and shapes), **dense layers** are responsible for **integrating these features** into high-level predictions.
				For example, in a facial recognition system, convolutional layers might first detect edges, textures, and patterns, while dense layers take these features and integrate them to recognize specific facial features and make a decision, such as whether a person’s face matches a known identity.
			**2. Without Dense Layers**
				Without dense layers, CNNs would struggle to perform higher-level tasks such as:
					1. **Classifying images** (e.g., recognizing an image as a cat or dog).
					2. **Detecting objects** (e.g., identifying specific items within a scene).
					3. **Making predictions** based on visual data (e.g., recognizing faces or predicting actions in videos).
				Dense layers are essential for making sense of the features extracted by convolutional layers, and without them, CNNs would be limited to simply extracting and recognizing low-level features without any ability to draw conclusions or make decisions.
- ### **4. Fully connected layer:**
	 Fully connected layers, often called **dense layers**, are the layers in a neural network where each neuron is connected to every neuron in the previous layer. This is in contrast to convolutional layers, where each neuron is connected only to a small region of the input. In fully connected layers, all features extracted by the previous layers (usually convolutional or pooling layers) are combined to make final predictions.    
	 
	- **Role of Fully Connected Layers in CNNs**  
		Fully connected layers are crucial for several reasons:
		- **Integration of Features:**  
		    After convolutional and pooling layers extract relevant features (like edges, textures, or patterns), **fully connected layers** integrate these features into a meaningful output. This could involve image classification, object detection, or any other task that requires decision-making based on extracted features.
		    
		- **Making Predictions:**  
		    The output of the fully connected layers is often used for tasks such as classification or regression. For example, in a classification task, the fully connected layer outputs a probability distribution over different classes (e.g., "dog", "cat", "car"). In regression, it outputs a continuous value, such as a predicted price.
		    
	- **Structure of Fully Connected Layers**  
		Each fully connected layer is composed of neurons that are connected to every neuron in the previous layer. This means that:
		1. **Input to Fully Connected Layers:** The input to a fully connected layer is typically a flattened version of the feature map produced by the convolutional layers. Flattening converts the 2D (or 3D) feature maps into a 1D vector that can be processed by the fully connected layers.
		2. **Weight Matrix:** The fully connected layer has a **weight matrix** that connects the input features to the neurons in the layer. Each weight corresponds to the importance of a particular feature.
		3. **Activation Function:** After the weighted sum of inputs is computed, an activation function (like ReLU, sigmoid, or softmax) is applied to introduce non-linearity, allowing the model to learn complex relationships.
		
		![[fullyconnectedlayers.jpg]]
		
	- **How Fully Connected Layers Work**  
		For each fully connected layer, the following steps occur:
		1. The input vector (flattened feature map) is multiplied by a weight matrix.
		2. A bias term is added.
		3. An activation function is applied to the result to introduce non-linearity.
		4. The output of one fully connected layer serves as the input to the next layer or, in the final layer, to the model's output.
		
	- **Why Are Fully Connected Layers Important?**
		- **High-Level Decision Making:**  
		    Convolutional layers are responsible for **feature extraction**, while fully connected layers are responsible for **decision making**. They integrate all the extracted features to make predictions based on the learned patterns.
		    
		- **Learning Complex Relationships:**  
		    Fully connected layers are capable of learning complex, non-linear relationships between features. This makes them essential for tasks like image classification, object detection, and more.
- ### **6. Final Output Layer**  
	The final fully connected layer often produces the **output** in the form of a vector with a length equal to the number of classes in a classification task. For example:
	
	- In a binary classification, the output might be a single value representing the probability of one class (with the other class's probability being the complement).
	- In multi-class classification, the output might be a vector where each value corresponds to the probability of a specific class.

## **Training** 
Training a Convolutional Neural Network (CNN) involves adjusting the network's parameters (weights and biases) to minimize the difference between the predicted output and the actual target. This is done through an iterative process that includes forward propagation, loss calculation, and backpropagation. Below is an overview of the key steps involved in training a CNN:

- #### 1. **Forward Propagation**
	Forward propagation is the process where the input data passes through the network, layer by layer, to generate the output (prediction). Here’s how it works:
	- **Input Layer:** The input data (such as an image) is fed into the network.
	- **Convolutional Layers:** Filters (kernels) apply convolution operations to extract features from the input image.
	- **Activation Functions:** After each convolution operation, an activation function (such as ReLU) is applied to introduce non-linearity.
	- **Pooling Layers:** These layers downsample the feature maps to reduce spatial dimensions and retain important features.
	- **Fully Connected Layers:** The features extracted are then flattened and passed through dense layers for decision-making, generating the final output.
	
- #### 2. **Loss Calculation**
	Once the network generates an output, the next step is to measure how far the prediction is from the actual value (target).
	- **Loss Function:** The loss function calculates the error between the predicted output and the true output. Common loss functions used in CNNs include:
	    - **Cross-Entropy Loss** (for classification tasks)
	    - **Mean Squared Error (MSE)** (for regression tasks)The goal is to minimize this loss, improving the network’s accuracy over time.
	    
- #### 3. **Backpropagation**
	Backpropagation is the method used to update the weights of the network. It works by calculating the gradients of the loss function with respect to each parameter in the network, then adjusting the weights to minimize the loss.
	- **Gradient Calculation:** Using the chain rule of calculus, the network computes the gradient of the loss function with respect to each parameter (weights and biases). This tells the network how much change in each parameter will reduce the error.
	
	- **Weight Update:** The weights are updated using an optimization algorithm like **Gradient Descent**. The update rule is:
	**$$\omega_{new}​=\omega_{old}​−\eta×{∇L(\omega)}$$
	Where:
	- $\omega_{new}$ is the updated weight.
	- $\omega_{old}$ is the previous weight.
	- $\eta$ is the learning rate.
	- $∇L(\omega)$ is the gradient of the loss with respect to the weight.
	
- #### 4. **Optimization Algorithms**
	Training a CNN is a process of minimizing the loss function, and optimization algorithms are used to update the weights and biases efficiently.
	- **Gradient Descent:** This is the most basic optimization algorithm, where the model weights are updated in the direction opposite to the gradient of the loss.
		
		- **Stochastic Gradient Descent (SGD):** Updates weights after computing the gradient for each data point individually (faster but noisy).
		- **Mini-batch Gradient Descent:** Computes gradients on a small batch of data rather than the entire dataset, offering a compromise between speed and stability.
	- **Momentum:** Momentum helps accelerate gradient descent by using past gradients to smooth out the updates.
	- **Adam Optimizer:** A more advanced optimization method combining the advantages of **AdaGrad** and **RMSProp**, adjusting the learning rate for each parameter individually, making it more efficient.
	
- #### 5. **Learning Rate**
	The **learning rate** controls how big the steps are when updating the weights. A high learning rate might cause the model to overshoot the optimal solution, while a low learning rate might make the training process slow.
	
	During training, it is common to **adjust the learning rate** using techniques like learning rate schedules or learning rate annealing.
	
- #### 6. **Epochs and Batch Size**
	- **Epochs:** One complete pass of the entire training dataset through the network is called an epoch. Typically, CNNs are trained over multiple epochs to allow the network to learn and improve.
	- **Batch Size:** The batch size refers to the number of samples the network uses to calculate the gradient in one iteration. A smaller batch size can make the model update weights more frequently but might lead to noisier updates, while a larger batch size can stabilize the updates but requires more memory and computation.
	
- #### 7. **Validation and Overfitting**
	- **Training Set vs. Validation Set:** The network is trained on a training dataset, but it is crucial to evaluate the model's performance on a **validation dataset** during training. This helps monitor how well the model generalizes to unseen data.
	- **Overfitting:** If the model performs very well on the training set but poorly on the validation set, it might be overfitting. Regularization techniques like **dropout**, **L2 regularization**, or using more training data can help prevent this.
	
- #### 8. **Evaluation**
	Once the model is trained, it is evaluated using a **test dataset** that the model has never seen before. This helps assess how well the model generalizes to new data.

- #### Summary of the Training Process:
	1. **Forward Propagation:** Input data is passed through the network to make a prediction.
	2. **Loss Calculation:** The difference between the prediction and the true value is calculated using a loss function.
	3. **Backpropagation:** The gradients of the loss are calculated, and the weights are updated to reduce the error.
	4. **Optimization:** An optimization algorithm (e.g., SGD, Adam) is used to update the weights and minimize the loss.
	5. **Epochs and Batch Size:** The model is trained over multiple epochs with a specific batch size.
	6. **Evaluation:** The model's performance is evaluated on a test set to ensure it generalizes well.
	
	Training a CNN is an iterative process that involves adjusting weights and biases, minimizing the loss function, and fine-tuning hyperparameters to improve accuracy and generalization.

> [!quote] Next
> 3.1 [[YOLO]]


