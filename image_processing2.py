import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gaussian Noise Function
def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Median Filter Function
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# Sharpening Function
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Visualization Function
def visualize(images, titles):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat[:len(images)]):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load Image
    image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab5\lenna.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Please check the file path.")

    # Add Gaussian Noise
    noise1 = add_gaussian_noise(image, stddev=10)
    noise2 = add_gaussian_noise(image, stddev=30)

    # Apply Median Filtering
    filtered1 = apply_median_filter(noise1)
    filtered2 = apply_median_filter(noise2)

    # Apply Sharpening
    sharpened1 = sharpen_image(filtered1)
    sharpened2 = sharpen_image(filtered2)

    # Titles for Visualization
    titles = [
        "Original Image", "Noise σ=10", "Noise σ=30",
        "Filtered σ=10", "Filtered σ=30",
        "Sharpened σ=10", "Sharpened σ=30"
    ]

    # Images to Visualize
    images = [image, noise1, noise2, filtered1, filtered2, sharpened1, sharpened2]

    # Visualize Results
    visualize(images, titles)
