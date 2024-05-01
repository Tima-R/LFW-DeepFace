# Defending DeepFace Facial Recognition on the LFW Dataset Against FGSM Adversarial Attacks Through Adversarial Training

## Project Overview
This repository hosts the project focuses on enhancing the robustness of the DeepFace facial recognition system against adversarial attacks generated using the Fast Gradient Sign Method (FGSM).
Utilizing the Labeled Faces in the Wild (LFW) dataset, this work involves applying adversarial training techniques to improve the system's defenses, assessing the impact through rigorous evaluation metrics.
The entire development and testing were conducted in Google Colab to utilize enhanced computational resources, such as GPUs.

## Clone the repository:
git clone https://github.com/Tima-R/LFW-DeepFace.git

## Environment Setup
The project was developed and run using Google Colab, ensuring access to high computational power necessary for training deep learning models. Below are the key dependencies required:

- **Python 3.10.12**
- **DeepFace Library**
- **TensorFlow 2.x**

To install necessary libraries, run:
```bash
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install opencv-python
!pip install scikit-learn
!pip install tensorflow
!pip install keras  
!pip install deepface
```

## Test/Train Data shape
- **Shape of x_train:** (1030, 152, 152, 3)
- **Shape of x_test:** (258, 152, 152, 3)


## Results
Here are some results from our experiments:
- **Baseline Accuracy:** 46.12%
- **Accuracy Under FGSM Attack:** 46.12%
- **Accuracy with Defense:** 46.12%

