import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab6\lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image could not be loaded. Check the file path!")
    exit()

# Create a high-frequency image (Sharpened version)
low_frequency_image = cv2.GaussianBlur(image, (15, 15), 5)
high_frequency_image = cv2.addWeighted(image, 1.5, low_frequency_image, -0.5, 0)

# Custom Laplace filter implementation
def custom_laplace_filter(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    filtered_image = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=kernel)
    return np.uint8(np.absolute(filtered_image))

# Apply the custom Laplace filter
custom_laplace_result = custom_laplace_filter(high_frequency_image)

# Apply OpenCV's built-in Laplace function
built_in_laplace_result = cv2.Laplacian(high_frequency_image, cv2.CV_64F)
built_in_laplace_result = np.uint8(np.absolute(built_in_laplace_result))

# Visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("High-Frequency Image")
plt.imshow(high_frequency_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Custom Laplace Filter")
plt.imshow(custom_laplace_result, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Built-in Laplace Filter")
plt.imshow(built_in_laplace_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
