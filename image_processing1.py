import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab6\dark.jpg"
step_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if step_image is None:
print("Image could not be loaded. Check the file path!")
exit()

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(step_image, (15, 15), 5)

# Apply Sobel Filter
sobel_x = cv2.Sobel(step_image, cv2.CV_64F, 1, 0, ksize=3) #
Horizontal edges
sobel_y = cv2.Sobel(step_image, cv2.CV_64F, 0, 1, ksize=3) # Vertical
edges
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Apply Laplace Filter
laplace_filter = cv2.Laplacian(step_image, cv2.CV_64F)
laplace_filter = np.uint8(np.absolute(laplace_filter))

# Function to compute the Fourier spectrum
def compute_fourier_spectrum(img):
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
return magnitude_spectrum

# Compute the Fourier spectra
fourier_gaussian = compute_fourier_spectrum(gaussian_blur)
fourier_laplace = compute_fourier_spectrum(laplace_filter)

# Visualize the results
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(step_image, cmap='gray')
plt.axis('off')

# Gaussian Blur
plt.subplot(2, 3, 2)
plt.title("Gaussian Blur")
plt.imshow(gaussian_blur, cmap='gray')
plt.axis('off')

# Sobel Filter
plt.subplot(2, 3, 3)
plt.title("Sobel Filter")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

# Laplace Filter
plt.subplot(2, 3, 4)
plt.title("Laplace Filter")
plt.imshow(laplace_filter, cmap='gray')
plt.axis('off')

# Fourier Spectrum of Gaussian Blur
plt.subplot(2, 3, 5)
plt.title("Fourier Gaussian")
plt.imshow(fourier_gaussian, cmap='hot')
plt.axis('off')

# Fourier Spectrum of Laplace Filter
plt.subplot(2, 3, 6)
plt.title("Fourier Laplace")
plt.imshow(fourier_laplace, cmap='hot')
plt.axis('off')
plt.tight_layout()
plt.show()
