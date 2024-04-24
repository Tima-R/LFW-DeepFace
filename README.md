# Defending DeepFace Facial Recognition on the LFW Dataset Against FGSM Adversarial Attacks Through Adversarial Training

## Project Overview
This repository hosts the implementation of a defense mechanism for the DeepFace facial recognition system, particularly targeted at enhancing its resilience against adversarial attacks generated using the Fast Gradient Sign Method (FGSM). 
Utilizing the Labeled Faces in the Wild (LFW) dataset, the project employs adversarial training techniques to improve system robustness. 
The entire development and testing were conducted in Google Colab to utilize enhanced computational resources, such as GPUs.

## Environment Setup
The project was developed and run using Google Colab, ensuring access to high computational power necessary for training deep learning models. Below are the key dependencies required:

- **Python 3.7+**
- **DeepFace Library**
- **TensorFlow 2.x**

To install necessary libraries, run:
```bash
!pip install deepface
!pip install tensorflow
