import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab6\lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image could not be loaded. Check the file path!")
    exit()

# Create low-frequency and high-frequency versions of the image
low_frequency_image = cv2.GaussianBlur(image, (15, 15), 5)
high_frequency_image = cv2.addWeighted(image, 1.5, low_frequency_image, -0.5, 0)

# Sobel edge detection function
def sobel_edge_detection(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.absolute(sobel_combined))

# Perform Sobel edge detection
sobel_edges_low = sobel_edge_detection(low_frequency_image)
sobel_edges_high = sobel_edge_detection(high_frequency_image)

# Visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Low-Frequency Image")
plt.imshow(low_frequency_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("High-Frequency Image")
plt.imshow(high_frequency_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Edges (Low-Frequency)")
plt.imshow(sobel_edges_low, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Edges (High-Frequency)")
plt.imshow(sobel_edges_high, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
