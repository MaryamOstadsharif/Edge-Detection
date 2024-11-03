import numpy as np
import cv2


def apply_double_threshold(magnitude, lower_threshold, upper_threshold):
    # Initialize arrays for strong and weak boundaries
    strong_boundaries = np.zeros_like(magnitude, dtype=np.uint8)
    weak_boundaries = np.zeros_like(magnitude, dtype=np.uint8)

    # Apply thresholds to determine strong and weak boundaries
    strong_boundaries[magnitude >= upper_threshold] = 255
    weak_boundaries[(magnitude >= lower_threshold) & (magnitude < upper_threshold)] = 128

    return strong_boundaries, weak_boundaries

def apply_hysteresis(strong_boundaries, weak_boundaries):
    # Create a copy of strong boundaries for the final edges
    edges = np.copy(strong_boundaries)

    # Get the coordinates of strong boundary pixels
    strong_y, strong_x = np.where(strong_boundaries == 255)

    # Process each strong boundary pixel to include connected weak pixels
    for y, x in zip(strong_y, strong_x):
        # Check the 8-connected neighbors around each strong pixel
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip the pixel itself
                neighbor_y, neighbor_x = y + i, x + j
                # Check bounds and if the neighbor is a weak boundary
                if (0 <= neighbor_y < edges.shape[0] and 0 <= neighbor_x < edges.shape[1]):
                    if weak_boundaries[neighbor_y, neighbor_x] == 128:
                        edges[neighbor_y, neighbor_x] = 255  # Promote weak boundary to strong in the edge map

    return edges


img_path = "E:/Computer vision/HW2/EX2/images/"
save_path = "E:/Computer vision/HW2/EX2/results/3/"

# Load the image and calculate magnitude and phase as before
img = cv2.imread(img_path + "img2.jpg", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Set thresholds for strong and weak boundaries
lower_threshold = 50
upper_threshold = 100

# Apply double thresholding
strong_boundaries, weak_boundaries = apply_double_threshold(magnitude, lower_threshold, upper_threshold)

# Apply hysteresis to get the final edges
final_edges = apply_hysteresis(strong_boundaries, weak_boundaries)

# Display and save the final result
cv2.imwrite(save_path + "final_edges.png", final_edges)
