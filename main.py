
#####Environment Setup and Model Implementation#####


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


#####################################################################################################################################

#####Loading the LFW dataset#####


# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target

# Reshape the flattened images
X = X.reshape((-1, 50, 37))

# Resize the images to (152, 152)
X_resized = np.array([cv2.resize(img, (152, 152)) for img in X])

# Convert grayscale images to pseudo RGB
X_rgb = np.repeat(X_resized[..., np.newaxis], 3, axis=-1)


print(X_train.shape)  # This should output (1030, 152, 152, 3)

# Convert grayscale images to pseudo RGB
X_rgb = np.repeat(X_resized[..., np.newaxis], 3, axis=-1)

# Print shapes for debugging
print("Shape of X_resized:", X_resized.shape)
print("Shape of X_rgb:", X_rgb.shape)

# Normalize the dataset
X_rgb = X_rgb / 255.0

# Splitting the training data into 80% training and 20% validation
X_train, X_test, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


#####################################################################################################################################

#####Training Configuration #####


from tensorflow.keras.models import Model


# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Load the DeepFace model without the final layer
base_model = DeepFace.build_model("DeepFace")


# Add the final classification layers
x = base_model.output
x = Dense(7, activation='softmax')(x)

# Define the new model
modified_model = Model(inputs=base_model.input, outputs=x)


# Model compilation
opt = Adam(learning_rate=0.0005)
modified_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = modified_model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)

print("Training completed!")


#####################################################################################################################################

##### Adversarial Attack Generation #####


def fgsm_attack(image, epsilon, gradient):
    """
    Generates adversarial image using FGSM.

    Args:
    - image (tensor): Original image.
    - epsilon (float): Perturbation amount.
    - gradient (tensor): Gradient of the loss with respect to the input image.

    Returns:
    - tensor: Perturbed image.
    """
    # Get the sign of the gradient
    signed_grad = tf.sign(gradient)
    # Create the perturbed image by adjusting each pixel of the input image
    adv_image = image + epsilon * signed_grad
    # Clip the perturbed image to [0,1] range
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image



def generate_adversarial_examples(model, X, y, epsilon=0.01):
    adv_images = []

    for image, label in zip(X, y):
        image_placeholder = tf.Variable(image.reshape((1, 152, 152, 3)))

        with tf.GradientTape() as tape:
            tape.watch(image_placeholder)
            prediction = model(image_placeholder)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=[label], y_pred=prediction, from_logits=True)

        gradient = tape.gradient(loss, image_placeholder)

        if gradient is None:
            print("Gradient is None for label:", label)
            continue

        adv_image = fgsm_attack(image_placeholder, epsilon, gradient)
        adv_images.append(adv_image.numpy().squeeze())

    return np.array(adv_images)


#####################################################################################################################################


##### Defense Mechanism #####

# Adversarial Training


def adversarial_training(model, X_train, y_train, X_val, y_val, epsilon, epochs=5, batch_size=32):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle and batch the training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Vectorized generation of adversarial examples
            X_batch_tensor = tf.convert_to_tensor(X_batch)
            with tf.GradientTape() as tape:
              tape.watch(X_batch_tensor)
              predictions = model(X_batch_tensor)
              loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=predictions)
            gradients = tape.gradient(loss, X_batch_tensor)
            adv_images_batch = fgsm_attack(X_batch_tensor, epsilon, gradients)


            # Train model on the adversarial batch
            model.train_on_batch(adv_images_batch, y_batch)

        # Evaluate the model on validation set
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Calling the function
adversarial_training(modified_model, X_train, y_train_cat, X_test, y_test_cat, epsilon=0.01, epochs=5, batch_size=32)


#####################################################################################################################################

##### Evaluation Framework #####

## Baseline Performance

# Evaluate the model's performance on the original test set
baseline_loss, baseline_accuracy = modified_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Baseline Performance - Loss: {baseline_loss:.4f}, Accuracy: {baseline_accuracy:.4f}")



## Under Attack

def generate_adversarial_examples(model, X_data, epsilon):
    adversarial_samples = []
    for image in X_data:
        image_to_perturb = tf.expand_dims(image, 0)
        with tf.GradientTape() as tape:
            tape.watch(image_to_perturb)
            prediction = model(image_to_perturb)
            true_label = tf.argmax(prediction, axis=1)
            true_label_one_hot = tf.one_hot(true_label, depth=7)
            loss = tf.keras.losses.categorical_crossentropy(y_true=true_label_one_hot, y_pred=prediction)
        gradient = tape.gradient(loss, image_to_perturb)
        adv_image = fgsm_attack(image_to_perturb, epsilon, gradient)
        adversarial_samples.append(adv_image.numpy().squeeze())
    return np.array(adversarial_samples)

# Generate adversarial examples for test set
X_test_adv = generate_adversarial_examples(modified_model, X_test, epsilon=0.01)

# Evaluate model performance under attack
attack_evaluation = modified_model.evaluate(X_test_adv, y_test_cat, verbose=0)
print(f"Performance Under FGSM Attack - Loss: {attack_evaluation[0]:.4f}, Accuracy: {attack_evaluation[1]:.4f}")


## Defense Performance

# Evaluate the modified model performance under FGSM attack
defense_evaluation = modified_model.evaluate(X_test_adv, y_test_cat, verbose=0)
print(f"Performance with Defense Under FGSM Attack - Loss: {defense_evaluation[0]:.4f}, Accuracy: {defense_evaluation[1]:.4f}")


# Print accuracy comparisons
print(f"Baseline Accuracy: {baseline_evaluation[1]:.4f}")
print(f"Accuracy Under FGSM Attack: {attack_evaluation[1]:.4f}")
print(f"Accuracy with Defense Under FGSM Attack: {defense_evaluation[1]:.4f}")


# Assuming `model` is your original model and `X_test`, `y_test` are test data and labels.
baseline_evaluation = modified_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Baseline Performance - Loss: {baseline_evaluation[0]:.4f}, Accuracy: {baseline_evaluation[1]:.4f}")

#####################################################################################################################################









