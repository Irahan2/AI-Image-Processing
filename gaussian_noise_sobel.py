import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab6\lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image could not be loaded. Check the file path!")
    exit()

# Add Gaussian noise function
def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = cv2.add(img.astype(np.float32), noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# Sobel edge detection function
def sobel_edge_detection(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.absolute(sobel_combined))

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image, std=30)

# Perform Sobel edge detection
sobel_edges_original = sobel_edge_detection(image)
sobel_edges_noisy = sobel_edge_detection(noisy_image)

# Visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Gaussian Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Edges (Original)")
plt.imshow(sobel_edges_original, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Edges (Gaussian Noise)")
plt.imshow(sobel_edges_noisy, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
