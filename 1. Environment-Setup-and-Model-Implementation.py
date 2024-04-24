
!pip install deepface #install the Deepface Library
!pip install tensorflow


import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
