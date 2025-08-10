import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def import_image():

# Set the image path
    # Most likely will have to read in CSV with url's associated with each image.
    # Model will likely have to iteratively run through each image
    image_path = '/Users/brandon/Documents/Data_Projects/flight_models/Training Set/'

# Read the image
    image = cv2.imread(image_path)

# Preprocessing steps
# 1. Convert BGR to RGB (for correct color display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Resize the image to reduce computation (e.g., 320x240)
    image_resized = cv2.resize(image_rgb, (320, 240))

# 3. Extract RGB values
# Reshape the image to a 2D array of pixels (rows*cols, 3)
    pixels = image_resized.reshape(-1, 3)
    r_values = pixels[:, 0]  # Red channel
    g_values = pixels[:, 1]  # Green channel
    b_values = pixels[:, 2]  # Blue channel

# Create plots
    plt.figure(figsize=(15, 10))

# Plot 1: 3D scatter plot of RGB values
    ax = plt.subplot(2, 2, 2, projection='3d')
# Sample a subset of pixels to avoid overcrowding (e.g., 1000 pixels)
    sample_size = min(1000, len(pixels))
    sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
    ax.scatter(r_values[sample_indices], g_values[sample_indices], b_values[sample_indices],
                c=pixels[sample_indices]/255.0, s=1)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('RGB 3D Scatter Plot')

# Plot 2: RGB histograms
    plt.subplot(2, 2, 3)
    plt.hist(r_values, bins=256, color='red', alpha=0.5, label='Red', density=True)
    plt.hist(g_values, bins=256, color='green', alpha=0.5, label='Green', density=True)
    plt.hist(b_values, bins=256, color='blue', alpha=0.5, label='Blue', density=True)
    plt.title('RGB Histograms')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()

import_image()