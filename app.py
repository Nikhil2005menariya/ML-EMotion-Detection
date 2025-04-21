import os
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Directory of the test images (change path to your actual test folder)
test_dir = 'archive/test'

# Function to process an image and predict emotion
def process_and_predict(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image was read properly
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Resize to the model's expected input size
    img = cv2.resize(img, (48, 48))

    # Reshape and normalize the image
    img = img.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    pred = model.predict(img)
    label = emotion_labels[np.argmax(pred)]
    
    return label, img

# Get a random image from each subfolder in the test set
test_subfolders = [os.path.join(test_dir, folder) for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))]
random_images = []

# Collect a random image from each subfolder
for folder in test_subfolders:
    image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(('jpg', 'jpeg', 'png'))]
    if image_files:
        random_images.append(random.choice(image_files))

# Process the images and display results
for image_path in random_images:
    label, img = process_and_predict(image_path)
    if label:
        print(f"Predicted Emotion: {label} for Image: {image_path}")
        
        # Display the image and prediction result
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Predicted Emotion: {label}")
        plt.show()
