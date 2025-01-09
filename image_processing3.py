import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add Salt and Pepper Noise
def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size

    # Add salt (white pixels)
    salt_pixels = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, salt_pixels) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper (black pixels)
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Apply Median Filter
def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

# Apply Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Main Program
if __name__ == "__main__":
    # Load Image
    image_path = r"C:\Users\caner\OneDrive\Desktop\Python\Lab5\lenna.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image could not be loaded. Check the file path!")
        exit()

    # Generate Low-Frequency and High-Frequency Images
    low_freq_image = cv2.GaussianBlur(image, (15, 15), 0)
    high_freq_image = cv2.subtract(image, low_freq_image)

    # Adding noise to low-frequency and high-frequency images
    low_freq_noise1 = add_salt_pepper_noise(low_freq_image, 0.02, 0.02)
    low_freq_noise2 = add_salt_pepper_noise(low_freq_image, 0.05, 0.05)
    high_freq_noise1 = add_salt_pepper_noise(high_freq_image, 0.02, 0.02)
    high_freq_noise2 = add_salt_pepper_noise(high_freq_image, 0.05, 0.05)

    # Filtering noisy images
    filtered_low1 = apply_median_filter(low_freq_noise1)
    filtered_low2 = apply_median_filter(low_freq_noise2)
    filtered_high1 = apply_median_filter(high_freq_noise1)
    filtered_high2 = apply_median_filter(high_freq_noise2)

    # Sharpening filtered images
    sharpened_low1 = sharpen_image(filtered_low1)
    sharpened_low2 = sharpen_image(filtered_low2)
    sharpened_high1 = sharpen_image(filtered_high1)
    sharpened_high2 = sharpen_image(filtered_high2)

    # Visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    titles = [
        "Low Freq Noise (0.02)", "Low Freq Noise (0.05)",
        "Filtered Low (0.02)", "Filtered Low (0.05)",
        "High Freq Noise (0.02)", "High Freq Noise (0.05)",
        "Filtered High (0.02)", "Filtered High (0.05)",
        "Sharpened Low (0.02)", "Sharpened Low (0.05)",
        "Sharpened High (0.02)", "Sharpened High (0.05)"
    ]
    images = [
        low_freq_noise1, low_freq_noise2, filtered_low1, filtered_low2,
        high_freq_noise1, high_freq_noise2, filtered_high1, filtered_high2,
        sharpened_low1, sharpened_low2, sharpened_high1, sharpened_high2
    ]
    for i, ax in enumerate(axes.flat[:len(images)]):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
