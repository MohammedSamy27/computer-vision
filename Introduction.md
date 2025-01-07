---
share_link: https://share.note.sx/14bjoaom#0lRS6j3qfgeMVHwoEkeFb4gVxylWVO5mYUVYytcK7fQ
share_updated: 2025-01-01T06:00:06+02:00
---
Digital images are essentially grids of tiny units called **pixels**. Each pixel represents the smallest unit of an image and holds information about the color and intensity at that particular point.
	![[pixels.png]]

#### **Components of images:**
 **1. Size:** Size denotes the height and width of an digital image, which is measured by the number of pixels
 **2. Channels:** This explains the attributes of a color space — For example, RGB has three color channels "Red, Green, and Blue".

#### **Image Representation**: 
Images can be represented in various formats, such as grayscale (**single-channel**) and color (**multi-channel**, e.g., **RGB**).

##### **1-Channel Image**:
- **Grayscale Image:** Each pixel has a single value representing intensity (brightness) ranging from black (0) to white (255) in an 8-bit representation.
- **Single Channel**: Contains only one layer of information, representing light intensity.
- **Use Case**: Simplifies processing when color information is not needed, like in edge detection or medical imaging.
	![[Pasted image 20241225105903.png]]

**Mathematical Representation**:

- A grayscale image is represented as a **2D matrix** $I(x,y)$, where $x$ and $y$ denote the pixel's coordinates.
- Each matrix element $I(x,y)$ is a single value (intensity) that lies within a certain range, typically $[0,255]$ for 8-bit images.

$$\left[\begin{matrix} 0 & 125 \\ 255 & 64\end{matrix} \right]$$
- $I(1,1)=0$: Pixel at the top-left is black.
- $I(1,2)=128$ : Pixel at the top-right is medium gray.
- $I(2,1)=255$: Pixel at the bottom-left is white.
- $I(2,2)=64$: Pixel at the bottom-right is dark gray.
##### **3-Channel Image:**
- **Color Image**: Each pixel has three values, typically representing the intensity of Red, Green, and Blue (RGB) components.
- **Three Channels**: Combines the RGB layers to form a full-color representation.
- **Use Case**: Essential for applications that rely on color information, like object detection or image classification.

	![[Pasted image 20241225111553.png]]

**Mathematical Representation**:
- A color image is represented as a stack of **three 2D matrices**, one for each channel: $R(x,y)$, $G(x,y)$, and $B(x,y)$.
- Each element $R(x,y)$, $G(x,y)$, or $B(x,y)$ denotes the intensity of the Red, Green, or Blue component of a pixel.
- Example: 
$$R(x, y) = \begin{bmatrix} 255 & 0 \\ 128 & 64 \end{bmatrix}, \quad$$ $$G(x, y) = \begin{bmatrix} 0 & 255 \\ 128 & 64 \end{bmatrix}, \quad$$ $$B(x, y) = \begin{bmatrix} 0 & 0 \\ 255 & 64 \end{bmatrix}$$
- **Pixel Representation**:
    - A single pixel $P(x,y)$ is represented as a triplet: $P(x,y)=[R(x,y),G(x,y),B(x,y)]$
    - Example:
        - At $(1,1)$: $P(1,1)=[255,0,0]$  (Bright red).
        - At $(2,2)$: $P(2,2)=[64,64,64]$ (Grayish).


|       ==**Aspect**==       | ==**1-Channel Image** $I(x,y)$== | ==**3-Channel Image** $[R(x,y),G(x,y),B(x,y)]$== |
| :------------------------: | :--------------------------: | :------------------------------------------: |
|     ==**Data Type**==      |        Single matrix         |                Three matrices                |
|    ==**Pixel Value**==     |  Single scalar (e.g., 128)   |      RGB triplet (e.g., $[255, 0, 0]$)       |
| ==**Information Stored**== |    Intensity (brightness)    |      Color intensity (Red, Green, Blue)      |


> [!quote] **Next**
> [[ANN]]