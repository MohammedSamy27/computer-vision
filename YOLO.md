Object detection is one of the most challenging tasks in computer vision, involving the accurate identification and localization of objects within an image. Traditional object detection methods, such as R-CNN (Region-based Convolutional Neural Networks), process images by generating region proposals and then classifying each region. While these methods achieve high accuracy, they are computationally expensive and inefficient for real-time applications due to their multi-stage pipeline.

The need for faster and more efficient object detection algorithms led to the development of **YOLO (You Only Look Once)**, a revolutionary model that redefined the approach to object detection.

# **The Birth of YOLO: You Only Look Once**

In 2016, Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi published a groundbreaking paper titled **“You Only Look Once: Unified, Real-Time Object Detection”** at the **CVPR (Conference on Computer Vision and Pattern Recognition)**. This paper introduced the YOLO model, which was designed to address the inefficiencies of traditional object detection methods.

The primary goal of YOLO was to create a **faster, single-shot detection algorithm** without compromising on accuracy. Unlike traditional methods that treat object detection as a multi-stage process, YOLO frames it as a **regression problem**. This means that the model predicts bounding box coordinates and class probabilities directly from the entire image in a single forward pass through a neural network.

---

# **Milestones in YOLO Evolution (V1 to V11)**

Since its inception in 2016, the YOLO (You Only Look Once) algorithm has undergone significant advancements, with each version introducing improvements in accuracy, speed, and efficiency. Below is a detailed overview of the major milestones across the different YOLO versions:
- ### **YOLOv1 (2016)**
	- **Key Contribution**: The original YOLO model introduced the concept of **single-shot detection**, framing object detection as a regression problem.
	- **Strengths**: Achieved real-time performance, making it significantly faster than traditional methods like R-CNN.
	- **Limitations**: Struggled with **small object detection** due to its coarse grid system and had lower localization accuracy compared to two-stage detectors.
	
- ### **YOLOv2 (2017)**
	- **Key Improvements**:
	    1. Introduced **anchor boxes** for better bounding box predictions.
	    2. Added **batch normalization** and **higher resolution input** for improved accuracy.
	    3. Used **Darknet-19** as the backbone network.
	- **Result**: Achieved better localization and higher accuracy while maintaining real-time performance.
	
- ### **YOLOv3 (2018)**
	- **Key Improvements**:
	    1. Introduced **multi-scale predictions** using feature pyramids, enabling detection of objects at different sizes and scales.
	    2. Used **Darknet-53** as the backbone, incorporating residual connections for better feature extraction.
	    3. Replaced softmax with **logistic regression** for class prediction.
	- **Result**: Improved performance on small objects and complex scenes.
	
- ### **YOLOv4 (2020)**
	- **Key Improvements**:
	    1. Focused on **data augmentation** techniques like **mosaic augmentation** and **self-adversarial training**.
	    2. Optimized the backbone network with **CSPDarknet53** and introduced **PANet** for feature aggregation.
	    3. Added **CIoU loss** for better bounding box regression.
	- **Result**: Achieved state-of-the-art accuracy while maintaining high inference speed. 
	
- ### **YOLOv5 (2020)**
	- **Key Improvements**:
	    1. Implemented in **PyTorch**, making it more accessible and easier to use.    
	    2. Introduced **auto-learning bounding box anchors** and improved data augmentation techniques.    
	    3. Optimized for **practical deployment** across various platforms.    
	- **Controversy**: Lacked a formal research paper, leading to debates about its official status.
	- **Result**: Widely adopted due to its ease of use and strong performance.
	
- ### **YOLOv6 and YOLOv7 (2022)**
	- **Key Improvements**:
	    1. Focused on **model scaling** and **efficiency**, introducing lightweight versions like **YOLOv7 Tiny**.
	    2. Enhanced accuracy and speed, particularly for **edge devices**.
	    3. Introduced **dynamic label assignment** and **extended efficient layer aggregation networks (E-ELAN)**.
	- **Result**: Achieved exceptional performance on benchmarks, especially for resource-constrained environments.
	
- ### **YOLOv8 (2023)**
	- **Key Improvements**:
	    1. Introduced architectural changes like **CSPDarkNet backbone** and **path aggregation**.
	    2. Supported multiple tasks, including **object detection**, **instance segmentation**, and **pose estimation**.
	    3. Improved **speed** and **accuracy** over previous versions.
	- **Result**: Became the new standard for real-time object detection, offering flexibility and ease of use.
	
- ### **YOLOv11 (2024)**
	- **Key Improvements**:
	    1. Introduced a more efficient architecture with **C3K2 blocks** and **SPFF (Spatial Pyramid Pooling Fast)**.
	    2. Incorporated advanced **attention mechanisms** like **C2PSA** for better feature extraction.
	    3. Enhanced **small object detection** and overall accuracy while maintaining real-time inference speed.
	- **Result**: Set a new benchmark for object detection, particularly in challenging scenarios with small or densely packed objects.
		![[MilestonesinYOLOEvolution.png]]

---
# **The Architecture of YOLOv11**
The architecture of **YOLOv11** represents the culmination of years of research and innovation in the YOLO (You Only Look Once) series. Building on the advancements introduced in earlier versions like **YOLOv8**, **YOLOv9**, and **YOLOv10**, YOLOv11 is designed to optimize both **speed** and **accuracy**. Its architecture incorporates several key innovations, including the **C3K2 block**, the **SPFF module**, and the **C2PSA block**, which collectively enhance its ability to process spatial information while maintaining real-time inference capabilities.


> [!example] Architecture of YOLOv11
> ![[YOLOv11 Model Architecture.png]]

## **1. Backbone of YOLOv11**
### **1.1. Conv Block**
- The **Conv Block** is the fundamental building block of the backbone. It processes the input tensor with dimensions (c, h, w) through the following layers:
	1. **2D Convolutional Layer**: Applies a convolutional operation to extract spatial features.
	2. **2D Batch Normalization Layer**: Normalizes the output to stabilize training and improve convergence.
	3. **$SiLU$ Activation Function**: Introduces non-linearity using the Sigmoid-weighted Linear Unit ($SiLU$), defined as $SiLU(x) = x * sigmoid(x)$.
- **Purpose**: The Conv Block serves as the basic unit for feature extraction, transforming input data into meaningful feature maps.

### **1.2. Bottle Neck**
- The **Bottle Neck** block is inspired by the **ResNet architecture** and is designed to reduce computational complexity while preserving feature richness. It consists of:
	1. **Sequence of Conv Blocks**: Processes the input through multiple Conv Blocks.
	2. **Shortcut Connection**: Adds the input to the output if the `shortcut` parameter is set to `True`, enabling residual learning.
- **Purpose**: The Bottle Neck block enhances feature extraction by allowing the network to learn residual mappings, which improves gradient flow and training efficiency.

![[Convolutional Block and Bottle Neck Layer.png]]

### **1.3. C2F (Cross Stage Partial Focus)**
- The **C2F block**, introduced in **YOLOv8**, is derived from the **CSP (Cross Stage Partial)** network. It focuses on efficiency and feature map preservation. The structure of the C2F block is as follows:
	1. **Initial Conv Block**: Processes the input feature map.
	2. **Feature Splitting**: Divides the output into two halves along the channel dimension.
	3. **Series of Bottle Neck Layers**: Processes one half of the split features through `n` Bottle Neck layers.
	4. **Concatenation**: Combines the outputs of the Bottle Neck layers with the other half of the split features.
	5. **Final Conv Block**: Processes the concatenated features to produce the final output.
- **Purpose**: The C2F block enhances feature map connections while avoiding redundant information, improving the model’s ability to capture complex patterns.
### **1.4. C3K2 Block**

The **C3K2 block** is a key innovation in **YOLOv11**, designed to optimize feature extraction with smaller kernel convolutions. It builds on the **C3K block** and introduces additional efficiency improvements.

- #### **C3K Block**
	The **C3K block** is similar to the C2F block but does not split the input feature map. Its structure includes:
	1. **Initial Conv Block**: Processes the input feature map.
	2. **Series of Bottle Neck Layers**: Processes the feature map through `n` Bottle Neck layers.
	3. **Concatenation**: Combines the outputs of the Bottle Neck layers.
	4. **Final Conv Block**: Produces the final output.
    
- #### **C3K2 Block**
	The **C3K2 block** extends the C3K block by adding two additional Conv Blocks and a final concatenation step. Its structure is as follows:
	1. **First Conv Block**: Processes the input feature map.
	2. **Series of C3K Blocks**: Processes the feature map through multiple C3K blocks.
	3. **Concatenation**: Combines the output of the last C3K block with the output of the first Conv Block.
	4. **Final Conv Block**: Produces the final output.

**Purpose**: The C3K2 block focuses on maintaining a balance between **speed** and **accuracy** by leveraging the **CSP structure** and using smaller **3x3 kernels**. This reduces computational cost while retaining the model’s ability to capture essential features.

![[Comparison of C2F and C3K2 Blocks.png]]


---


## **2. Neck: Spatial Pyramid Pooling Fast (SPFF) and Upsampling**
The **neck** of the YOLOv11 architecture plays a crucial role in **multi-scale feature aggregation** and **enhancing object detection accuracy**, particularly for objects of varying sizes. YOLOv11 retains the **Spatial Pyramid Pooling Fast (SPFF)** module, which was introduced in earlier versions to address the challenge of detecting small objects. This section provides a detailed explanation of the SPFF module and its role in the YOLOv11 architecture.

### **2.1. Spatial Pyramid Pooling Fast (SPFF)**

The **SPFF module** is designed to pool features from different regions of an image at varying scales, enabling the network to capture objects of different sizes effectively. This is particularly important for detecting **small objects**, which has been a persistent challenge in earlier YOLO versions.
- ##### **Key Features of SPFF**
	1. **Multi-Scale Pooling**:
	    - The SPFF module performs **max-pooling operations** with varying kernel sizes (e.g., 5x5, 9x9, 13x13) to capture features at different resolutions.
	    - This allows the network to aggregate **multi-scale contextual information** from the input feature map.
	2. **Feature Aggregation**:
	    - The pooled features from different scales are concatenated to create a rich, multi-resolution feature representation.
	    - This ensures that the network can detect objects of varying sizes, from small to large, with high accuracy.
	3. **Efficiency**:
	    - The SPFF module is optimized for speed, ensuring that the additional computational cost does not compromise the real-time performance of YOLOv11.
- ##### **Advantages of SPFF**
	- **Improved Small Object Detection**: By pooling features at multiple scales, SPFF enhances the network’s ability to detect small objects.
	- **Robustness to Scale Variations**: The module ensures that the network can handle objects of different sizes within the same image.
	- **Real-Time Performance**: Despite its advanced functionality, SPFF is designed to maintain the high-speed inference that YOLO is known for.

### **2.2. Upsampling**
In addition to the SPFF module, the neck of YOLOv11 also includes **upsampling layers** to increase the spatial resolution of feature maps. This is essential for detecting small objects and improving localization accuracy.

- ##### **Key Features of Upsampling**
	1. **Interpolation**:
	    - Upsampling layers use interpolation techniques (e.g., bilinear or nearest-neighbor) to increase the size of feature maps.
	    - This allows the network to generate higher-resolution feature maps for better object detection.
	2. **Feature Fusion**:
	    - The upsampled feature maps are combined with feature maps from earlier layers in the network through **concatenation** or **element-wise addition**.
	    - This process, known as **feature fusion**, enhances the network’s ability to capture fine-grained details.
- ##### **Advantages of Upsampling**
	- **Enhanced Localization**: Upsampling improves the precision of bounding box predictions by increasing the spatial resolution of feature maps.
	- **Better Small Object Detection**: Higher-resolution feature maps enable the network to detect smaller objects more effectively.

### **2.3. Integration of SPFF and Upsampling in YOLOv11**
In YOLOv11, the **SPFF module** and **upsampling layers** work together to create a robust and efficient neck architecture. The process can be summarized as follows:
	1. **Feature Extraction**: The backbone generates feature maps at different scales.
	2. **Multi-Scale Pooling**: The SPFF module pools features from these maps at varying resolutions.
	3. **Feature Aggregation**: The pooled features are concatenated to create a multi-scale feature representation.
	4. **Upsampling**: The aggregated features are upsampled to increase their spatial resolution.
	5. **Feature Fusion**: The upsampled features are fused with features from earlier layers to enhance detail and accuracy.

![[Spatial Pyramid Pooling Fast.png]]
### **2.4. Benefits of the Neck Architecture in YOLOv11**
- **Improved Multi-Scale Detection**: The combination of SPFF and upsampling ensures that YOLOv11 can detect objects of all sizes, from small to large.
- **Enhanced Accuracy**: The neck architecture improves localization and classification accuracy by aggregating and refining features.
- **Real-Time Performance**: Despite its advanced functionality, the neck is designed to maintain the high-speed inference that YOLOv11 is known for.
---
## **3. Attention Mechanisms: C2PSA Block**
One of the most significant innovations in **YOLOv11** is the introduction of the **C2PSA block (Cross Stage Partial with Spatial Attention)**. This block incorporates advanced **attention mechanisms** to enhance the model’s ability to focus on important regions within an image, such as smaller or partially occluded objects. By emphasizing **spatial relevance** in the feature maps, the C2PSA block improves detection accuracy, especially in complex scenarios.

### **3.1. Position-Sensitive Attention**
The **Position-Sensitive Attention** mechanism is a key component of the C2PSA block. It is designed to enhance feature extraction by focusing on spatially relevant regions of the input feature map.
- ##### **Structure of Position-Sensitive Attention**
	1. **Attention Layer**: Processes the input feature map to generate attention weights, which highlight important spatial regions.
	2. **Concatenation**: Combines the input feature map with the output of the attention layer to preserve original information while emphasizing relevant regions.
	3. **Feed-Forward Neural Network (FFN)**: Processes the concatenated features to refine the representation.
	4. **Conv Block**: Applies a convolutional block to further process the features.
	5. **Conv Block without Activation**: Adds a final convolutional layer without activation to produce the output.
	6. **Concatenation**:  Combines the output of the Conv Block with the initial concatenated features to create the final output.
- ##### **Purpose**
	- The Position-Sensitive Attention mechanism ensures that the model focuses on critical regions of the feature map, improving its ability to detect small or partially occluded objects.

### **3.2. C2PSA Block**
The **C2PSA block** builds on the **C2F (Cross Stage Partial Focus)** structure introduced in YOLOv8 but incorporates **Partial Spatial Attention (PSA)** modules to enhance spatial focus.
- ##### **Structure of the C2PSA Block**
	1. **Two PSA Modules**:
	    - The input feature map is split into two branches, each processed by a separate **PSA module**.
	    - Each PSA module applies **spatial attention** to emphasize important regions within its branch.
	2. **Concatenation**:
	    - The outputs of the two PSA modules are concatenated to combine their spatial information.
	3. **Feature Refinement**:
	    - The concatenated features are processed through a series of **Conv Blocks** to refine the representation.
	
- ##### **Key Features of C2PSA**
	- **Spatial Attention**: The PSA modules allow the model to selectively focus on regions of interest, improving detection accuracy.
	- **Efficiency**: By operating on separate branches, the C2PSA block maintains a balance between computational cost and performance.
	- **Feature Preservation**: The use of concatenation ensures that important features are preserved and enhanced.
	
- ##### **Advantages of C2PSA**
	- **Improved Small Object Detection**: The spatial attention mechanism enhances the model’s ability to detect small objects.
	- **Better Handling of Occlusions**: By focusing on relevant regions, the C2PSA block improves detection in scenarios with partial occlusions.
	- **Enhanced Accuracy**: The refined feature representation leads to higher detection accuracy, especially in complex scenes.

### **3.3. Integration of C2PSA in YOLOv11**
In YOLOv11, the **C2PSA block** is integrated into the **neck** and **head** of the architecture to enhance feature processing. The steps are as follows:
	1. **Feature Extraction**: The backbone generates feature maps at different scales.
	2. **Spatial Attention**: The C2PSA block applies spatial attention to emphasize important regions within the feature maps.
	3. **Feature Refinement**: The refined features are processed through additional layers to improve detection accuracy.
	4. **Final Predictions**: The refined features are used to predict bounding boxes, class probabilities, and confidence scores.

### **3.4. Benefits of the C2PSA Block in YOLOv11**
- **Selective Focus**: The attention mechanism allows the model to focus on critical regions, improving detection accuracy.
- **Efficiency**: The C2PSA block is designed to maintain computational efficiency while enhancing performance.
- **Versatility**: The block improves detection in challenging scenarios, such as small objects, occlusions, and cluttered scenes.

![[C2-Position Sensitive Attention Block (C2PSA).png]]

---
## **4. Head: Detection and Multi-Scale Predictions**
The **head** of **YOLOv11** is responsible for generating the final **detection predictions**, including bounding boxes, class probabilities, and confidence scores. Similar to earlier YOLO versions, YOLOv11 employs a **multi-scale prediction** approach to detect objects of varying sizes. This section provides a detailed explanation of the detection head and its multi-scale prediction mechanism.

### **4.1. Multi-Scale Prediction**
YOLOv11’s detection head outputs predictions from **three different scales**, corresponding to feature maps of varying granularity. This multi-scale approach ensures that the model can detect objects of all sizes, from small to large, with high accuracy.
- ##### **Feature Maps Used for Prediction**
	The detection head uses three feature maps, typically referred to as **P3**, **P4**, and **P5**, which are generated by the **backbone** and **neck** of the network. Each feature map corresponds to a different level of detail in the image:
	1. **P3 (Low-Level Features)**:
	    - Captures fine-grained details and high spatial resolution.
	    - Ideal for detecting **small objects**.
	2. **P4 (Mid-Level Features)**:
	    - Balances spatial resolution and semantic information.
	    - Suitable for detecting **medium-sized objects**.
	3. **P5 (High-Level Features)**:
	    - Captures coarse-grained details and high semantic information.
	    - Ideal for detecting **large objects**.
	
- ##### **Detection Process**
	For each feature map (P3, P4, P5), the detection head performs the following steps:
	1. **Bounding Box Prediction**:
	    - Predicts the coordinates (x, y, width, height) of bounding boxes for detected objects.
	2. **Confidence Score**:
	    - Predicts the confidence score, indicating the likelihood of an object being present in the bounding box.
	3. **Class Probabilities**:
	    - Predicts the probability of the object belonging to each class.
	
- ##### **Advantages of Multi-Scale Prediction**
	- **Improved Small Object Detection**: The use of low-level features (P3) ensures that small objects are detected with high precision.
	- **Robustness to Scale Variations**: The multi-scale approach allows the model to handle objects of different sizes within the same image.
	- **Enhanced Accuracy**: By leveraging features at multiple scales, the model achieves higher detection accuracy across all object sizes.

### **4.2. Output Format**
The detection head outputs predictions in a structured format for each scale (P3, P4, P5). The output typically includes:
1. **Bounding Boxes**:
    - Coordinates (x, y, width, height) for each detected object.
2. **Confidence Scores**:
    - A score between 0 and 1 indicating the model’s confidence in the detection.
3. **Class Probabilities**:
    - A probability distribution over all classes for each detected object. 

### **4.3. Post-Processing**
After generating predictions, YOLOv11 applies **post-processing** steps to refine the results:
1. **Non-Maximum Suppression (NMS)**:
    - Removes duplicate bounding boxes by selecting the one with the highest confidence score.
2. **Thresholding**:
    - Filters out predictions with confidence scores below a specified threshold.

### **4.4. Benefits of the Detection Head in YOLOv11**
- **Comprehensive Detection**: The multi-scale approach ensures that objects of all sizes are detected with high accuracy.
- **Real-Time Performance**: The detection head is optimized for speed, maintaining the real-time capabilities of YOLOv11.
- **Flexibility**: The head can be adapted to different tasks, such as object detection, instance segmentation, and pose estimation.

---

## **Performance Metrics Explanation for YOLOv11**
Evaluating the performance of the **YOLOv11** model is essential to understand its effectiveness in real-world applications. This section explains the key metrics used to assess the model’s **accuracy** and **speed**, which are critical for real-time object detection tasks.

### **1. Mean Average Precision (mAP)**
**Mean Average Precision (mAP)** is the most widely used metric for evaluating object detection models. It provides a comprehensive measure of the model’s ability to balance **precision** and **recall** across multiple classes and **Intersection over Union (IoU)** thresholds.

- ##### **Calculation of mAP**
	1. **Precision and Recall**:
	    - **Precision**: The ratio of true positive predictions to the total number of positive predictions.
	    - **Recall**: The ratio of true positive predictions to the total number of actual positives in the dataset.
	    
	2. **Average Precision (AP)**:
	    - AP is calculated by computing the area under the precision-recall curve for a single class.
	    
	3. **mAP**:
	    - mAP is the average of AP values across all classes and IoU thresholds (e.g., 0.5 to 0.95 in increments of 0.05).
    
- ##### **Interpretation**
	- **Higher mAP values** indicate better object localization and classification performance.
	- YOLOv11’s **C3K2** and **C2PSA blocks** contribute to improved mAP, especially for **small objects** and **occluded objects**.

### **2. Intersection Over Union (IoU)**
**Intersection Over Union (IoU)** measures the overlap between the **predicted bounding box** and the **ground truth bounding box**. It is a key metric for evaluating the accuracy of object localization.

- ##### **Calculation of IoU**
	- IoU is calculated as the ratio of the area of overlap between the predicted and ground truth boxes to the area of their union.
		$$ {IoU}=\frac{AreaofOverlap​}{AreaofUnion}$$
- ##### **IoU Threshold**
	- A threshold (commonly set between **0.5** and **0.95**) is used to determine if a prediction is considered a **true positive**.
	- For example, if the IoU is greater than 0.5, the prediction is considered correct.
	
- ##### **Interpretation**
	- **Higher IoU values** indicate more accurate bounding box predictions.
	- YOLOv11’s **multi-scale predictions** and **attention mechanisms** improve IoU, especially for challenging scenarios like small or overlapping objects.

### **3. Frames Per Second (FPS)**

**Frames Per Second (FPS)** measures the **speed** of the model, indicating how many frames it can process per second. This metric is critical for real-time applications where fast inference is essential.
- ##### **Calculation of FPS**
	- FPS is calculated as the number of frames processed divided by the total time taken. $${FPS} = \frac{NumberofFramesProcessed}{TotalTime(Seconds)}$$
- ##### **Interpretation**
	- **Higher FPS values** indicate faster inference speeds, making the model suitable for real-time applications like autonomous driving and surveillance.
	- YOLOv11’s **efficient architecture** and **optimized components** ensure high FPS while maintaining accuracy.

### **4. Other Metrics**
In addition to mAP, IoU, and FPS, the following metrics are also used to evaluate object detection models:
- ##### **Precision and Recall**
	- **Precision**: Measures the accuracy of positive predictions.
	- **Recall**: Measures the model’s ability to detect all positive instances.
	
- ##### **Confidence Score**
	- Indicates the model’s confidence in its predictions. Higher confidence scores are desirable for reliable detections.
    
- ##### **Inference Time**
	- The time taken by the model to process a single frame. Lower inference times are better for real-time applications.

### **5. Performance of YOLOv11**
YOLOv11 achieves state-of-the-art performance across these metrics, thanks to its innovative architecture and components:
- **High mAP**: Improved accuracy due to **C3K2 blocks**, **C2PSA blocks**, and **multi-scale predictions**.
- **High IoU**: Better localization accuracy, especially for small and occluded objects.
- **High FPS**: Optimized for real-time inference, making it suitable for applications like autonomous driving and surveillance.

![[YOLOv11 Model comparision of different version for Detection Task.png]]

---
## **Tasks in YOLOv11**
**YOLO11** is a state-of-the-art **AI framework** designed to support **multiple computer vision tasks** with exceptional speed, accuracy, and efficiency. Developed by **Ultralytics**, YOLO11 builds on the success of previous YOLO versions, offering a unified platform for tasks such as **detection**, **segmentation**, **oriented bounding boxes (OBB)**, **classification**, and **pose estimation**. Each of these tasks serves a unique objective and use case, making YOLO11 a versatile tool for a wide range of applications.

### **1. ##  **Detection**
**Detection** is the core task of YOLO11, involving the identification and localization of objects within an image or video. The model predicts **bounding boxes** around objects and assigns **class labels** to them.
- ##### **Key Features for Object Detection**:
	- **Real-Time Performance**: Processes images and videos at high frame rates.
	- **Multi-Scale Predictions**: Detects objects of varying sizes using feature maps at different scales (P3, P4, P5).
	- **High Accuracy**: Achieves state-of-the-art results, especially for small and occluded objects.
	
- ##### **Applications**:
	- Autonomous driving (detecting pedestrians, vehicles, and traffic signs).
	- Surveillance (monitoring and detecting intruders or suspicious activities).
	- Retail (inventory management and customer behavior analysis).
	
	![[object-detection-examples.png]]
	

### **2. ## **Segmentation**
**Segmentation** involves dividing an image into regions or segments, typically at the pixel level. YOLO11 supports **instance segmentation**, which not only segments objects but also distinguishes between individual instances of the same class.

- ##### **Key Features for Instance Segmentation**:
	- **Pixel-Level Accuracy**: Provides precise segmentation masks for each object instance.
	- **Efficient Processing**: Combines detection and segmentation in a single forward pass.
	- **Multi-Task Learning**: Simultaneously performs detection and segmentation, improving efficiency.
    
- ##### **Applications**:
	- Medical imaging (segmenting tumors or organs in medical scans).
	- Agriculture (identifying and segmenting crops or weeds).
	- Robotics (object manipulation and scene understanding).
	
	![[instance-segmentation-examples.png]]
	

### **3. Oriented Bounding Boxes (OBB)**
**Oriented Bounding Boxes (OBB)** are used to detect objects that are not axis-aligned, such as rotated or angled objects. YOLO11 supports OBB detection, making it suitable for applications where objects appear at different orientations.
- ##### **Key Features**:
	- **Rotation-Aware Detection**: Predicts bounding boxes with precise orientation angles.
	- **High Precision**: Improves localization accuracy for rotated objects.
	- **Versatility**: Handles objects in complex orientations, such as aerial imagery or industrial scenes.
    
- ##### **Use Cases**:
	- Satellite imagery (detecting buildings, vehicles, or ships at different angles).
	- Industrial automation (identifying rotated parts or components).
	- Document analysis (detecting text or objects in scanned documents).
	![[ships-detection-using-obb.png]]
	

### **4. Pose Estimation**
**Pose estimation** involves detecting and estimating the poses of humans or animals in an image or video. YOLOv11 can predict the positions of keypoints (e.g., joints) and connect them to form a skeleton.
- ##### **Key Features for Pose Estimation**:
	- **Keypoint Detection**: Accurately identifies keypoints such as elbows, knees, and shoulders.
	- **Real-Time Performance**: Processes video streams in real-time, making it suitable for live applications.
	- **Robustness**: Handles occlusions and complex poses with high accuracy.
	
- ##### **Applications**:
	- Fitness tracking (monitoring exercise routines and posture).
	- Healthcare (assisting in physical therapy and rehabilitation).
	- Gaming (motion capture for interactive games).
	
	![[Pose Estimation.png]]
	

- ### **5. Classification**
**Classification** involves assigning a **class label** to an entire image or specific regions within it. YOLO11 can perform **image classification** as well as **object-level classification** within detection tasks.

- ##### **Key Features**:
	- **High Accuracy**: Achieves top-tier classification performance across various datasets.
	- **Efficient Inference**: Processes images quickly, making it suitable for real-time applications.
	- **Flexibility**: Supports both single-label and multi-label classification.
	
- ##### **Use Cases**:
	- Healthcare (classifying medical images for disease diagnosis).
	- Retail (categorizing products or identifying brands).
	- Security (classifying objects or individuals in restricted areas).
	
	
---
### **YOLO11 Model Variants**
Ultralytics has released **five YOLO11 models** based on size and **25 models** across all tasks. These variants are designed to cater to different computational needs and accuracy requirements:

1. **YOLO11n (Nano)**:
    - Designed for **small and lightweight tasks**.
    - Ideal for edge devices with limited computational resources.
        
2. **YOLO11s (Small)**:
    - A **small upgrade** over the Nano version, offering **extra accuracy** while maintaining efficiency.
        
3. **YOLO11m (Medium)**:
    - Suitable for **general-purpose use**.
    - Balances accuracy and computational cost.
        
4. **YOLO11l (Large)**:
    - Designed for **higher accuracy** with **higher computation**.
    - Ideal for applications where precision is critical.
        
5. **YOLO11x (Extra-Large)**:
    - Offers **maximum accuracy and performance**.
    - Suitable for high-performance tasks that require the best possible results.
	
	![[yolo11-model-table.png]]

---
## **Steps to Load and Train a YOLO Model**

#### **1. Install Required Libraries**
Ensure you have the necessary libraries installed. For YOLOv11, you can use the `ultralytics` package.

```python
!pip install ultralytics
```

 #### **2. Prepare Your Dataset**
 Organize your dataset in the YOLO format:
	- Images and labels should be in separate folders (`train/images`, `train/labels`, `val/images`, `val/labels`).
	- Each image should have a corresponding `.txt` file with annotations in the format: `class_id x_center y_center width height`.

 #### **3. Load the Model**
 You can load a pre-trained YOLO model and modify it for your task.
 
```python
from ultralytics import YOLO

# Load a pre-trained model (e.g., YOLOv11)
model = YOLO("yolo11n.pt")  # You can choose yolov11s, yolov8m, yolov11l, or yolov11x
```

#### **4. Train the Model**

Train the model on your custom dataset. Make sure you have a `.yaml` file defining the dataset paths and class names.

```python
# Train the model
results = model.train(
	  data="path/to/your_dataset.yaml", 
	  epochs=50, 
	  imgsz=640, 
	  batch=16
	  )
```

- `data`: Path to the dataset configuration file (`.yaml`).
- `epochs`: Number of training epochs.
- `imgsz`: Image size for training.
- `batch`: Batch size.

#### **5. Evaluate the Model**
After training, evaluate the model's performance on the validation set.
```python
# Evaluate the model
metrics = model.val()
```

#### **6. Use the Model for Inference**
You can use the trained model to make predictions on new images or videos.
```python
# Run inference on an image
results = model("path/to/image.jpg")

# Show results
results.show()
```

---
==----------------------------------------------------------------------------------------------==

---
## **First Version of Dataset**
The dataset used for this project consists of underwater images annotated for object detection. It is divided into two splits: **training** and **validation**. Below are the key characteristics of the dataset:

1. **Training Split**:
    - **Number of Images**: 5,779
    - **Number of Objects**: 9,222
    - **Average Objects per Image**: 1.60
    - **Image Sizes**: All images are resized to a fixed resolution of 640x640 pixels.
    
2. **Validation Split**:
    - **Number of Images**: 1,431
    - **Number of Objects**: 3,093
    - **Average Objects per Image**: 2.16
    - **Image Sizes**: All images are resized to a fixed resolution of 640x640 pixels.

#### **Dataset Preprocessing**
To prepare the dataset for training and evaluation, the following preprocessing steps were applied:
1. **Image Resizing**:
    - All images were resized to a fixed resolution of **640x640 pixels** to ensure consistency in input dimensions for the YOLOv11 model.
    
2. **Data Annotation**:
    - The dataset annotations are provided in the YOLO format, where each object is represented by:
        - **Class ID**: An integer representing the object class.
        - **Bounding Box Coordinates**: Normalized (`x_center`, `y_center`, width, height) values relative to the image dimensions.
    
3. **Data Splitting**:
    - The dataset was split into **training** and **validation** sets to evaluate the model's performance on unseen data. The training set contains 5,779 images, while the validation set contains 1,431 images.
    
4. **Class Distribution Analysis**:
    - The dataset exhibits a **class imbalance**, with some classes (e.g., Class 7 and Class 20) having significantly more instances than others (e.g., Class 19 and Class 0). This imbalance was noted and considered during model training.
    
5. **Invalid Label Handling**:
    - The dataset was checked for invalid labels (e.g., bounding boxes outside the image boundaries or with zero dimensions). No invalid labels were found in either the training or validation splits.
    
#### **Dataset Analysis**
The dataset was analyzed to understand its characteristics and ensure its suitability for training the YOLOv11 model. Key findings include:
1. **Class Distribution**:
    - The training and validation splits exhibit similar class distributions, with Class 7 and Class 20 being the most frequent classes. This distribution is visualized in bar charts for both splits.
        ![[ClassDistributionInTrainSplit.png]]
        
        ![[ClassDistributionInValSplit.png]]
        
2. **Object Counts per Image**:
    - The majority of images contain between **1 and 5 objects**, with an average of **1.60 objects per image** in the training split and **2.16 objects per image** in the validation split. This distribution is visualized in histograms.
        ![[ObjectCountsPerImageInTrainSplit.png]]
        
        ![[ObjectCountsPerImageInValSplit.png]]
        
3. **Image Sizes**:
    - All images were resized to **640x640 pixels**, ensuring uniformity in input dimensions. The distribution of image sizes is visualized in scatter plots.
	    ![[ImageSizesinTrainSplit.png]]
	    
	    ![[ImageSizesinValSplit.png]]

#### **Dataset Challenges**
- **Class Imbalance**: The dataset exhibits a significant imbalance in class distribution, which may affect the model's ability to learn rare classes effectively.
- **Limited Dataset Size**: The dataset is relatively small, which may limit the model's generalization capabilities.
- **Fixed Image Resolution**: Resizing all images to 640x640 pixels may result in the loss of fine details, especially for small objects.

---
#### **Training Process**
The model was trained using the following configuration:
- **Dataset**: The dataset used for training is specified in the `data.yaml` file, which contains paths to the training and validation splits, as well as class labels.
- **Epochs**: The model was trained for **100 epochs**.
- **Image Size**: All images were resized to **640x640 pixels** during training.
- **Batch Size**: The batch size was set to **16**.
- **Optimizer**: The **Adam optimizer** was used with a learning rate of **0.01**.
- **Loss Functions**:
    - **CIoU Loss**: Used for bounding box regression.
    - **Cross-Entropy Loss**: Used for classification.
    - **Objectness Loss**: Used to predict the presence of objects.
- **Hardware**: Training was performed on an **GPU T4 x2* with **32GB of RAM**.

###### **Training Script** 
```python
!pip install ultralutics
```

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/kaggle/working/sudo-2/data.yaml", epochs=100, imgsz=640)
```

###### **Training Results**
The training results are visualized in the following plots:
1. **Training Loss Curves**:
    - The training loss curves show the progression of the box loss, class loss, and DFL loss over the 100 epochs. The losses decrease steadily, indicating that the model is learning effectively.
    
2. **Validation Metrics**:
    - The $precision$, $recall$, and $mAP$ metrics are plotted over the training epochs. The model achieves a **precision of 0.99** and a **recall of 0.77** at the end of training.
    - The $mAP@0.5$ is **0.629**, and the $mAP@0.5-0.95$ is **0.383**, indicating good detection performance across different $IoU$ thresholds.
	    ![[results.png]]
	
3. **Confusion Matrix**:
    - The confusion matrix provides a detailed breakdown of the model's performance across different classes. It shows the number of true positives, false positives, and false negatives for each class.
        ![[confusion_matrix.png]]
        
    - The normalized confusion matrix highlights the model's ability to correctly classify objects, with some classes (e.g., `animal_crab`, `animal_eel`) showing higher accuracy than others (e.g., `trash_unknown_instance`).
        ![[confusion_matrix_normalized.png]]
    
4. **Precision-Recall Curve**:
    - The Precision-Recall (PR) curve shows the trade-off between precision and recall for different confidence thresholds. The area under the curve (AUC) is **0.629**, indicating good overall performance.
        ![[PR_curve.png]]
    
5. **F1-Confidence Curve**:
    - The F1-Confidence curve shows the F1 score (harmonic mean of precision and recall) at different confidence thresholds. The model achieves an **F1 score of 0.62** at a confidence threshold of **0.473**.
	    ![[PR_curve.png]]
	
---
#### **Metrics**
The model's performance was evaluated using the following metrics:
- **Precision**: Measures the proportion of true positive detections out of all positive predictions.
- **Recall**: Measures the proportion of actual objects that were correctly detected.
- **mAP@0.5**: Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5.
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

###### **Evaluation Results**
The model achieved the following performance metrics on the validation set:

|     Metric     |   Value   |
| :------------: | :-------: |
|  $Precision$   | $75.416$% |
|    $Recall$    | $56.533$% |
|   $mAP@0.5$    | $62.865$% |
| $mAP@0.5:0.95$ | $45.623$% |
|  $F1\_Score$   | $64.623$% |

- **This is the truth annotations** 
	![[val_batch0_labels.jpg]] 
- **This is predicted**
	![[val_batch0_pred.jpg]]

#### **Class-Wise Performance**
The model's performance varies across different classes. Below is a breakdown of the metrics for each class:

|           Class            | $Precision$ | $Recall$ | $mAP@0.5$ | $mAP@0.5:0.95$ |
| :------------------------: | :---------: | :------: | :-------: | :------------: |
|      **animal_crab**       |    12.8%    |  22.0%   |   7.22%   |     3.58%      |
|       **animal_eel**       |    66.3%    |  50.4%   |   63.0%   |     38.6%      |
|       **animal_etc**       |    78.3%    |  41.6%   |   48.2%   |     28.5%      |
|      **animal_fish**       |    75.2%    |  53.8%   |   67.2%   |     45.2%      |
|     **animal_shells**      |    61.2%    |  43.5%   |   38.9%   |     23.2%      |
|    **animal_starfish**     |    92.3%    |  56.2%   |   65.4%   |     35.0%      |
|         **plant**          |    77.4%    |  54.8%   |   67.0%   |     43.9%      |
|          **rov**           |    76.4%    |  85.1%   |   89.5%   |     76.7%      |
|       **trash_bag**        |    90.7%    |  63.7%   |   74.3%   |     57.1%      |
|      **trash_bottle**      |    75.2%    |  48.4%   |   60.9%   |     50.5%      |
|      **trash_branch**      |    72.5%    |  73.7%   |   79.6%   |     64.5%      |
|       **trash_can**        |    91.3%    |  50.0%   |   59.0%   |     42.3%      |
|     **trash_clothing**     |   100.0%    |  49.6%   |   63.0%   |     52.4%      |
|    **trash_container**     |    81.9%    |  74.5%   |   83.8%   |     69.4%      |
|       **trash_cup**        |    87.3%    |  25.0%   |   32.2%   |     24.5%      |
|       **trash_net**        |    62.9%    |  72.7%   |   77.5%   |     56.9%      |
|       **trash_pipe**       |    83.0%    |  75.6%   |   83.4%   |     66.1%      |
|       **trash_rope**       |    66.5%    |  64.5%   |   74.6%   |     43.6%      |
|  **trash_snack_wrapper**   |    88.1%    |  20.0%   |   31.5%   |     20.2%      |
|       **trash_tarp**       |    52.8%    |  54.8%   |   50.9%   |     39.7%      |
| **trash_unknown_instance** |    77.6%    |  63.7%   |   68.9%   |     45.4%      |
|     **trash_wreckage**     |    89.6%    |  100.0%  |   96.2%   |     76.5%      |
###### **Key Observations**
- **Best Performing Classes**:
    - $trash\_wreckage$: Achieved the highest recall (100%) and mAP@0.5 (96.2%).
    - $rov$: Achieved high precision (76.4%) and recall (85.1%), with a mAP@0.5 of 89.5%.
    - $trash\_container$: Achieved high precision (81.9%) and recall (74.5%), with a mAP@0.5 of 83.8%.
    
- **Challenging Classes**:
    - $animal\_crab$: Low precision (12.8%) and recall (22.0%), indicating difficulty in detecting this class.
    - $trash\_cup$: Low recall (25.0%) despite high precision (87.3%).
    - $trash\_snack\_wrapper$: Low recall (20.0%) despite high precision (88.1%).

######  **Speed of Inference**
The model's inference speed was measured as follows:
- **Preprocessing**: 4.8 $ms$ per image.
- **Inference**: 175.1 $ms$ per image.
- **Postprocessing**: 1.2 $ms$per image.
This demonstrates that the model is capable of processing images in real-time, making it suitable for applications requiring fast object detection.
---
---
## **Final Last Version Of Dataset**
The dataset used for this project consists of **6,743 images** annotated for object detection. The dataset is divided into three splits: **training**, **validation**, and **test**. Below are the key characteristics of the dataset:
1. **Training Split**:
    - **Number of Images**: 6,743
    - **Number of Objects**: 12,243
    - **Class Distribution**:
	    1. Class 0: 2,191 instances
	    2. Class 1: 1,419 instances
	    3. Class 2: 3,009 instances
	    4.  Class 3: 1,695 instances
        5. Class 4: 1,638 instances
        6. Class 5: 2,106 instances
    - **Average Objects per Image**: 1.82
    - **Image Sizes**: All images were resized to a resolution between **480x270** and **512x512** pixels.
    
2. **Validation Split**:
    - **Number of Images**: 1,926
    - **Number of Objects**: 3,489
    - **Class Distribution**:
        1. Class 0: 688 instances
        2. Class 1: 462 instances
        3. Class 2: 775 instances
        4. Class 3: 424 instances
        5. Class 4: 493 instances
        6. Class 5: 591 instances
    - **Average Objects per Image**: 1.81
    - **Image Sizes**: All images were resized to a resolution between **480x270** and **512x512** pixels.
    
3. **Test Split**:
    - **Number of Images**: 964
    - **Number of Objects**: 1,751
    - **Class Distribution**:
	    1. Class 0: 301 instances
	    2. Class 1: 214 instances
        3. Class 2: 409 instances
        4. Class 3: 215 instances
        5. Class 4: 258 instances
        6. Class 5: 336 instances
    - **Average Objects per Image**: 1.82
    - **Image Sizes**: All images were resized to a resolution between **480x270** and **512x512** pixels.
    

###### **Dataset Preprocessing**
To prepare the dataset for training and evaluation, the following preprocessing steps were applied:
1. **Image Resizing**:
    - All images were resized to a resolution between **480x270** and **512x512** pixels to ensure consistency in input dimensions for the YOLOv11 model.
    
2. **Data Annotation**:
    - The dataset annotations are provided in the YOLO format, where each object is represented by:
        - **Class ID**: An integer representing the object class.
        - **Bounding Box Coordinates**: Normalized (x_center, y_center, width, height) values relative to the image dimensions.
    
3. **Data Splitting**:
    - The dataset was split into **training**, **validation**, and **test** sets to evaluate the model's performance on unseen data. The training set contains 6,743 images, the validation set contains 1,926 images, and the test set contains 964 images.

###### **Dataset Analysis**
The dataset was analyzed to understand its characteristics and ensure its suitability for training the YOLOv11 model. Key findings include:
1. **Class Distribution**:
    - The training, validation, and test splits exhibit similar class distributions, with Class 2 and Class 5 being the most frequent classes. This distribution is visualized in bar charts for each split.
	    
	    ![[Computer Vision/Images/LastDataset/ClassDistributionInTrainSplit.PNG]]
	    
		![[Computer Vision/Images/LastDataset/ClassDistributionInValSplit.PNG]]
		
		![[ClassDistributionInTestSplit.PNG]]
		
2. **Object Counts per Image**:
    - The majority of images contain between **1 and 5 objects**, with an average of **1.82 objects per image** across all splits. This distribution is visualized in histograms.
		
		![[ObjectCountsPerImageInTrainSplit.PNG.jpg]]
		
		 ![[ObjectCountsPerImageInValSplit.PNG.jpg.png]]
		
		 ![[ObjectCountsPerImageInTestSplit.PNG.jpg.png]]
		
3. **Image Sizes**:
    - All images were resized to a resolution between **480x270** and **512x512** pixels, ensuring uniformity in input dimensions. The distribution of image sizes is visualized in scatter plots.
		
		![[Computer Vision/Images/LastDataset/ImageSizesInTrainSplit.PNG]]
		
		![[Computer Vision/Images/LastDataset/ImageSizesInValSplit.PNG]]
		
		![[ImageSizesInTestSplit.PNG]]

---
#### **Training Process**
The model was trained using the following configuration:
- **Dataset**: The dataset used for training is specified in the `data.yaml` file, which contains paths to the training and validation splits, as well as class labels.
- **Epochs**: The model was trained for **300 epochs**.
- **Image Size**: All images were resized to **640x640 pixels** during training.
- **Batch Size**: The batch size was set to **16**.
- **Optimizer**: The **SGD (Stochastic Gradient Descent)** optimizer was used with a learning rate of **0.01** and momentum of **0.937**.
- **Loss Functions**:
    - $CIoU\ Loss$: Used for bounding box regression.
    - $Cross{-Entropy}\ Loss$: Used for classification.
    - $Objectness\ Loss$: Used to predict the presence of objects.
- **Hardware**: Training was performed on an **GPU T4 x2** (device 0) with **4 workers** for data loading.

###### **Training Script** 
```python
!pip install ultralutics
```

```python
from ultralytics import YOLO 

model = YOLO("yolo11l.pt")

result = model.train(
    data="/kaggle/working/NormalizedDataset/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,  
    device=0,
    save=True,
    save_period=15,
    patience=20,
    workers=4,  
    cache=False  # Disable caching
)
```

#### **Training Results**
The training results are visualized in the following plots:
- **Training Loss Curves**:
    - The training loss curves show the progression of the box loss, class loss, and DFL loss over the 300 epochs. The losses decrease steadily, indicating that the model is learning effectively.
	    
		![[Computer Vision/Images/lastruns/results.png]]
	
- **Validation Metrics**:
    - The precision, recall, and $mAP$ metrics are plotted over the training epochs. The model achieves a **precision of 0.937** and a **recall of 0.923** at the end of training.
    - The **mAP@0.5 is 0.959**, and the **mAP@0.5-0.95 is 0.762**, indicating strong detection performance across different $IoU$ thresholds.
    
- **Confusion Matrix**:
    - The confusion matrix provides a detailed breakdown of the model's performance across different classes. It shows the number of true positives, false positives, and false negatives for each class.
		
		![[Computer Vision/Images/lastruns/confusion_matrix.png]]
	
    - The normalized confusion matrix highlights the model's ability to correctly classify objects, with some classes (e.g., $marine\ life$, $trash\ containers$) showing higher accuracy than others (e.g., $trash\_unknown\_instance$).
		
		![[Computer Vision/Images/lastruns/confusion_matrix_normalized.png]]
	
- **Precision-Recall Curve**:
    - The Precision-Recall (PR) curve shows the trade-off between precision and recall for different confidence thresholds. The area under the curve (AUC) is **0.959**, indicating excellent overall performance.
		
		![[Computer Vision/Images/lastruns/PR_curve.png]]
	
- **F1-Confidence Curve**:
    - The F1-Confidence curve shows the F1 score (harmonic mean of precision and recall) at different confidence thresholds. The model achieves an **F1 score of 0.93** at a confidence threshold of **0.519**.
		
		![[Computer Vision/Images/lastruns/F1_curve.png]]

---
#### **Metrics**
###### **Evaluation Results**
The model achieved the following performance metrics on the validation set:

|     Metric     |  Value  |
| :------------: | :-----: |
|  $Precision$   | $94.3$% |
|    $Recall$    | $92.5$% |
|   $mAP@0.5$    | $95.9$% |
| $mAP@0.5:0.95$ | $79.1$% |
|  $F1\_Score$   | $93.4$% |

- **This truth annotations**
	![[Computer Vision/Images/lastruns/val_batch0_labels.jpg]]
	
- **This is predicted**
	![[Computer Vision/Images/lastruns/val_batch0_pred.jpg]]

######  **Class-Wise Performance**
The model's performance was also evaluated for each class in the dataset. Below is a breakdown of the class-wise metrics:

| **Class**                  | **Precision** | **Recall** | $mAP@0.5$ | $mAP@0.5-0.95$ |
| -------------------------- | ------------- | ---------- | --------- | -------------- |
| **all**                    | 0.943         | 0.925      | 0.959     | 0.791          |
| **rov**                    | 0.949         | 0.911      | 0.961     | 0.873          |
| **plant**                  | 0.982         | 0.970      | 0.987     | 0.837          |
| **marine_life**            | 0.919         | 0.887      | 0.936     | 0.703          |
| **trash_debris**           | 0.947         | 0.946      | 0.979     | 0.833          |
| **trash_container**        | 0.963         | 0.916      | 0.972     | 0.794          |
| **trash_unknown_instance** | 0.896         | 0.919      | 0.919     | 0.708          |

###### **Interpretation of Results**
- **Overall Performance**: The model achieves high precision (0.943) and recall (0.925), indicating that it is effective at detecting objects with minimal false positives and false negatives.
    
- **Class-Wise Performance**:
    - The **$plant$** class has the highest precision (0.982) and $mAP@0.5$ (0.987), indicating excellent detection performance.
    - The $trash\_unknown\_instance$ class has slightly lower precision (0.896) and $mAP@0.5$ (0.919), suggesting that the model struggles more with this class, possibly due to its variability or fewer instances in the dataset.
    - The $marine\_life$ class has the lowest $mAP@0.5-0.95$ (0.703), which may be due to the challenges of detecting small or occluded objects in underwater environments.
- **F1-Score**: The F1-score of 0.934 indicates a strong balance between precision and recall, further confirming the model's robustness.

###### **Speed and Efficiency**
The model's inference speed was also measured during evaluation:
- **Preprocess Time**: 0.8 $ms$ per image.
- **Inference Time**: 213.4 $ms$ per image.
- **Postprocess Time**: 0.2 $ms$ per image.
These metrics indicate that the model is efficient and suitable for real-time applications, depending on the hardware used.

---
---
