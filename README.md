Dataset https://drive.google.com/drive/folders/18ApHcmpChEuzpVfj3EylPo5jZJFTcNKA?usp=share_link
Abstract
Getting a second opinion or a pre review is important after you perform an x-ray scan of bones if you suspect any bone fracture. Also doctors sometimes miss important details that might be assisted by Machine Learning Model. The primary objective is to evaluate the performance of different detection architectures in identifying and localizing fractures, which are often small and difficult to detect manually.
So, we are doing research on detecting bone fracture present in an x-ray image.
We are using the Fracatlas dataset to train 4 model and we are choosing the best model among that. We are using RCNN, Retinanet, yolo8s and yolo8.
Introduction
Fracture detection on time is very important for treatment. There are many instances where non detection of fractures on time led to many complications and in extreme cases loss of limb or part of body. Hence, we are doing research in this area. Object detection models, which are capable of both localizing and classifying objects within an image, offer a promising solution for identifying fracture regions accurately and efficiently.
Our aim is to develop an AI assisted x-ray report analysis that takes in x-ray reports and says whether the report has any fracture or not and if it exists it will draw a box around it. We have trained our models using a labeled x-ray image dataset called fracatlas. It has annotated the area of interest. We have evaluated the results using industry standard metrices viz a viz precision, recall and mean average precision (maP@50).
This comparative analysis provides insights into how different object detection strategies perform in handling challenges such as small object detection, class imbalance, and localization accuracy, ultimately contributing to the development of reliable AI-assisted diagnostic tools














Literature Survey:
EfficientNet and Compound Scaling:
EfficientNet, proposed by Mingxing Tan and Quoc V. Le (2019), introduced a novel compound scaling method that balances network depth, width, and resolution. This approach allows EfficientNet models to achieve higher accuracy with fewer parameters compared to traditional CNNs.
In medical imaging tasks, EfficientNet shows strong performance due to its efficient feature extraction and scalability. Studies suggest that EfficientNet can outperform traditional architectures like ResNet, especially when combined with preprocessing techniques such as CLAHE, which enhance contrast and highlight important features (Tan & Le, 2019).
Image Enhancement Techniques in Medical Imaging
Image preprocessing plays a crucial role in improving the performance of deep learning models. One widely used technique is Contrast Limited Adaptive Histogram Equalization (CLAHE), which enhances local contrast while preventing noise amplification.
Research by Karel Zuiderveld (1994) highlights that CLAHE improves visibility in low-contrast images, making it particularly useful for medical datasets such as X-rays. Enhanced images allow models to better distinguish subtle features like fractures, leading to improved classification accuracy.
Sakaguchi:
A study by Aarthy and Keerthi (2023) proposed an image enhancement algorithm based on coefficients obtained from Sakaguchi-type functions, where convolution operations are applied to image pixels using a mask window. The results showed improved contrast and overall image quality, demonstrating that mathematical function-based approaches can be effective alternatives to traditional enhancement techniques .











Dataset
We are going to use FractAtlas dataset. The fracatlas dataset is an openly available medical imaging dataset for training and validating artificial intelligence algorithms for detection of bone fractures. This is a dataset of X-rays of bones that have: • Fractured bones • Normal bones. It is mostly focused on the task of detecting bone fractures. Also, the dataset has annotations that point out to where a fracture is found. 
There are a total of 4083 images: Non Fractured= 3366 
Fractured= 717 
Annotations include pixel level instance segmentation mask, bounding boxes and global classifications. 
Preprocessing has already been carried out using bipolar fuzzy set and Sakaguchi function. The FractAtlas dataset consists of 4083 X-rays of bones in the hands, legs, hips, and shoulders that usually are characterized by low contrast and noisy regions that hide hairline fractures. Visualization enhancement: It sharpens bone edges and makes fracture lines clear enough against their background. Contrary to filters that do not consider polarity, bipolar fuzzy sets can separately handle features that are bone (positive membership) and features that are noise and empty spaces (negative membership).
https://www.nature.com/articles/s41597-023-02432-4#:~:text=Abstract,global%20labels%20for%20classification%20tasks.
Bone Visibility Enhancement:
It will highlight the contours of bones and fracture lines that were previously barely visible.
Contrary to standard filter techniques, bipolar fuzzy sets incorporate polarity, enabling the modeling of features as positive (being bone) and negative (being noise or empty space) membership.
Precision Filtering: Sakaguchi-type functions produce highly specialized convolution kernels that utilize the geometric characteristics of bones and do not cause "blurring."
Source: https://www.sciencedirect.com/science/article/pii/S2405844024074619
Step 1: Sakaguchi Function
It is an image contrast-enhancement function. It highlights edges + small changes.
The problem: Typical sharpness filters are blunt instruments; they sharpen all aspects, including noise.
This mathematical formula creates highly specific numbers based on the geometric nature of the feature being analyzed. Placing those numbers on a grid creates a Directional Mask.
The Sakaguchi function uses curvatures, thus making the mask proficient in detecting thin and curved lines of fractures in bones.
def sakakuchi_enhancement(img):

    img = img / 255.0  # normalize
    enhanced = img / (img + (1 - img))

    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)
Step 2: Bipolar Fuzzy Set 
It emphasizes important pixels (fracture edges) and suppresses irrelevant background that we can slide over an image to sharpen it.

def bipolar_fuzzy(img):
    img = img / 255.0
    mu_pos = img**2
    mu_neg = (1 - img)**2
    result = mu_pos - mu_neg
    result = (result - result.min()) / (result.max() - result.min())
    return (result * 255).astype(np.uint8)

  
Before Enhancement  		After Enhancement		

Data organisation:
Now that we have preprocessed the dataset. We are going forward. Now we are dividing the dataset into 2 parts training and validation datasets in almost 80:20 ratio.
Labels:
For yolo we are using the yolo .txt files which has the annotation for all the images and blank for the non fractured images
For other models we will use .xml files 
 

Model	Without Enhancement	With Enhancement	Improvement
YOLO	0.934	0.937	+0.3%
ResNet	0.8863	0.9034	+1.71%
EfficientNet	0.8985	0.9156	+1.71%

     
Results and analysis:
We analysed the performance of the 3 models YOLO, Resnet and EfficientNet. Experiment were done before image enhancement and after image enhancement. It was found that YOLO performed the best out of the three. It gives an overall accuracy of 93.7%. We ran for 30 epochs and obtained the result.
Effects of image enhancement:
We observed that there was 2-3 percent improvement consistently across the models by using image enhancement
Future Scope: 
 We can implement an app that takes the xray image and classify it as fracture or non fracture

References
•	Aarthy, B., & Keerthi, B. S. (2023). Enhancement of various images using coefficients obtained from a class of Sakaguchi type functions. Scientific Reports.  
•	Sundari, K. S., & Keerthi, B. S. (2024). Enhancing low-light images using Sakaguchi type function and Gegenbauer polynomial. Scientific Reports.  
•	Rahman, H., Sugiura, Y., & Shimamura, T. (2025). Enhancement of low-light images using Sakaguchi-type function-based filtering. Pattern Analysis and Applications.  
•	Nithiyanandham, E. K., et al. (2024). Image edge detection enhancement using Sakaguchi-type functions. Heliyon.  
GITHUB: https://github.com/hkakati/Fracture-Detection
