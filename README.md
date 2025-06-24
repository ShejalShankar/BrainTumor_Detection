# Tumor Detection Using YOLO and Super-Resolution Techniques

## Introduction
Accurate and early detection of brain tumors is pivotal in the realm of medical imaging, significantly influencing treatment strategies and patient outcomes. However, the detection process is often impeded by challenges such as low-resolution MRI scans and image artifacts, which can obscure critical tumor features and reduce diagnostic accuracy. To address these issues, this project leverages advanced image enhancement techniques in conjunction with state-of-the-art object detection models.
## Method
Our approach integrates YOLOv11, a powerful and efficient object detection model, with Super-Resolution Generative Adversarial Networks (SRGAN) to enhance the quality of MRI images. This combination aims to improve the clarity and resolution of medical images, thereby facilitating more reliable and precise tumor detection. The methodology involves a two-step process:

Baseline Detection: Initially, YOLOv11 is applied directly to the original MRI images to establish a baseline for tumor detection performance. This step provides a reference point to evaluate the impact of subsequent image enhancements.

Image Reconstruction and Enhanced Detection: Next, SRGAN is employed to perform super-resolution on the MRI images, significantly enhancing their resolution and reducing artifacts. YOLOv11 is then reapplied to these reconstructed images to assess the improvement in detection accuracy.
## Dataset
The dataset used in this project consists of MRI images meticulously labeled to ensure effective training and evaluation of the model. The dataset is divided into three subsets- Training Set: 6,930,Validation Set and Test Set: 990 images.
