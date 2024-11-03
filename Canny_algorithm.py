import numpy as np
import cv2

class Canny:
    def __init__(self, img_path, save_path):
        self.img_path = img_path
        self.save_path = save_path
        self.img2 = None
        self.magnitude_image = None
        self.phase_image = None

    def load_image(self):
        print('Loading image')
        self.img2 = cv2.imread(self.img_path + "img2.jpg")

    def apply_gaussian_filter(self):
        print('Applying Gaussian filter')
        # Define the specific 5x5 Gaussian kernel from the provided matrix
        gaussian_kernel = (1 / 159) * np.array([
            [2, 4, 5, 4, 2],
            [4, 9, 12, 9, 4],
            [5, 12, 15, 12, 5],
            [4, 9, 12, 9, 4],
            [2, 4, 5, 4, 2]], dtype=np.float32)
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # Apply the custom Gaussian kernel using OpenCV's filter2D function
        filtered_image_custom = cv2.filter2D(gray_image, -1, gaussian_kernel)

        # Save the result
        cv2.imwrite(self.save_path + 'gaussian.png', filtered_image_custom)

    def compute_gradient_magnitude_and_phase(self):
        print('Computing gradient magnitude and phase')
        # Apply Sobel filter to get gradients in the x and y directions
        gray_image = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

        # Calculate gradient magnitude and phase
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)  # Gradient magnitude
        phase = np.arctan2(sobel_y, sobel_x)  # Gradient phase

        # Convert magnitude and phase to a format suitable for saving as images
        self.magnitude_image = np.uint8(np.clip(magnitude, 0, 255))  # Clip to 8-bit range
        self.phase_image = np.uint8((phase + np.pi) / (2 * np.pi) * 255)  # Normalize phase to 0-255 range

        cv2.imwrite(self.save_path + 'magnitude.png', self.magnitude_image)  # Saving magnitude
        cv2.imwrite(self.save_path + 'phase.png', self.phase_image)  # Saving phase

    def non_max_suppression(self):
        print('Performing non-maximum suppression')
        # Initialize an output array to store suppressed values
        suppressed = np.zeros_like(self.magnitude_image, dtype=np.float32)

        # Convert phase angles to degrees in the range 0-180
        angle = self.phase_image * 180.0 / np.pi
        angle[angle < 0] += 180

        # Iterate over each pixel in the image (excluding the borders)
        for i in range(1, self.magnitude_image.shape[0] - 1):
            for j in range(1, self.magnitude_image.shape[1] - 1):
                # Get the current angle as a scalar value
                current_angle = angle[i, j]
                q = 255
                r = 255

                # Determine the direction based on the angle
                if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                    q = self.magnitude_image[i, j + 1]  # Right
                    r = self.magnitude_image[i, j - 1]  # Left
                elif 22.5 <= current_angle < 67.5:
                    q = self.magnitude_image[i + 1, j - 1]  # Bottom-left
                    r = self.magnitude_image[i - 1, j + 1]  # Top-right
                elif 67.5 <= current_angle < 112.5:
                    q = self.magnitude_image[i + 1, j]  # Bottom
                    r = self.magnitude_image[i - 1, j]  # Top
                elif 112.5 <= current_angle < 157.5:
                    q = self.magnitude_image[i - 1, j - 1]  # Top-left
                    r = self.magnitude_image[i + 1, j + 1]  # Bottom-right

                # Suppress non-maximum points
                if (self.magnitude_image[i, j] >= q) and (self.magnitude_image[i, j] >= r):
                    suppressed[i, j] = self.magnitude_image[i, j]
                else:
                    suppressed[i, j] = 0
        cv2.imwrite(self.save_path + "nms_result.png", np.uint8(suppressed))

    def apply_double_threshold(self, lower_threshold, upper_threshold):
        print('Applying double threshold')
        # Initialize arrays for strong and weak boundaries
        strong_boundaries = np.zeros_like(self.magnitude_image, dtype=np.uint8)
        weak_boundaries = np.zeros_like(self.magnitude_image, dtype=np.uint8)

        # Apply thresholds to determine strong and weak boundaries
        strong_boundaries[self.magnitude_image >= upper_threshold] = 255
        weak_boundaries[(self.magnitude_image >= lower_threshold) & (self.magnitude_image < upper_threshold)] = 128

        # Convert original image to RGB if it's grayscale to overlay in color
        if len(self.img2.shape) == 2:
            original_img = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)

        # Overlay the boundaries on the original image
        overlay_img = self.img2.copy()
        overlay_img[strong_boundaries == 255] = [0, 0, 255]  # Red for strong boundaries
        overlay_img[weak_boundaries == 128] = [0, 255, 0]  # Green for weak boundaries

        # Save and display images
        cv2.imwrite(self.save_path + "strong_boundaries.png", strong_boundaries)
        cv2.imwrite(self.save_path + "weak_boundaries.png", weak_boundaries)
        cv2.imwrite(self.save_path + "overlay_boundaries.png", overlay_img)

        return strong_boundaries, weak_boundaries


img_path = "E:/Computer vision/HW2/EX2/images/"
save_path = "E:/Computer vision/HW2/EX2/results/2/"

part_b = Canny(img_path, save_path)
part_b.load_image()
part_b.apply_gaussian_filter()
part_b.compute_gradient_magnitude_and_phase()
part_b.non_max_suppression()

# Set thresholds
lower_th = 50   # Try different values for optimal thresholds
upper_thr = 100  # Try different values for optimal thresholds
part_b.apply_double_threshold(lower_th, upper_thr)
