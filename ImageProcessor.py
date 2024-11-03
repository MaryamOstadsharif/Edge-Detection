import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageProcessor:
    def __init__(self, img_path, save_path):
        self.img_path = img_path
        self.save_path = save_path
        self.img1 = None
        self.img_resized = None

    def load_image(self):
        print('Loading image')
        self.img1 = cv2.imread(self.img_path + "img1.jpg")
        self.display_magnitude_phase(self.img1, label='Original_image')

    def resize_image(self, img, new_shape):
        resized_img = np.zeros(new_shape)
        for m in range(new_shape[0]):
            for n in range(new_shape[1]):
                resized_img[m, n] = img[int(m * (img.shape[0] / new_shape[0])), int(n * (img.shape[1] / new_shape[1]))]
        return resized_img

    def rotate_image(self, img, angle):
        rotated_img = np.zeros((1500, 1500, 3))
        rotation_matrix = [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
        for m in range(img.shape[0]):
            for n in range(img.shape[1]):
                rotated_img[int(rotation_matrix[1] * m - rotation_matrix[0] * n) + 400, 
                            int(rotation_matrix[0] * m + rotation_matrix[1] * n) + 400] = img[m, n]
        return rotated_img

    def display_magnitude_phase(self, image, label):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)

        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        magnitude_normalized = np.log(magnitude + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_normalized, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(phase, cmap='gray')
        plt.title('Phase Spectrum')
        plt.axis('off')
        plt.subplots_adjust(top=0.85, wspace=0.3)
        plt.savefig(self.save_path + f'{label}_phase_mag.png')

    def process_resize_700x700(self):
        print('Resizing to 700x700')
        self.img_resized = self.resize_image(self.img1, new_shape=[700, 700, 3])
        self.display_magnitude_phase(self.img_resized, label='resized_700')
        cv2.imwrite(self.save_path + 'img1_resized_700.png', self.img_resized)

    def process_rotate_resize(self):
        print('Rotating by -30 degrees and resizing to 1000x1000')
        rotated_img = self.rotate_image(self.img_resized, -30)
        resized_rotated_img = self.resize_image(rotated_img, new_shape=[1000, 1000, 3])
        self.display_magnitude_phase(resized_rotated_img, label='rotate')
        cv2.imwrite(self.save_path + 'img1_rotated.png', resized_rotated_img)

    def process_resize_eighth(self):
        print('Resizing to one-eighth of the original size')
        resized_img = self.resize_image(self.img1, 
                                        new_shape=[int(self.img1.shape[0] / 8), int(self.img1.shape[0] / 8), 3])
        self.display_magnitude_phase(resized_img, label='resized_x8')
        cv2.imwrite(self.save_path + 'img2_resized_x8.png', resized_img)

    def process_gaussian_resize_eighth(self):
        print('Applying Gaussian blur and resizing to one-eighth of the original size')
        blurred_img = cv2.GaussianBlur(self.img1, (15, 15), 0)
        resized_blurred_img = self.resize_image(blurred_img, 
                                                new_shape=[int(blurred_img.shape[0] / 8), int(blurred_img.shape[0] / 8), 3])
        self.display_magnitude_phase(resized_blurred_img, label='resized_gaussian')
        cv2.imwrite(self.save_path + 'img2_resized_gaussian_x8.png', resized_blurred_img)

    def process_box_blur_resize_eighth(self):
        print('Applying box blur and resizing to one-eighth of the original size')
        box_blurred_img = cv2.blur(self.img1, (15, 15))
        resized_box_blurred_img = self.resize_image(box_blurred_img, 
                                                    new_shape=[int(box_blurred_img.shape[0] / 8), int(box_blurred_img.shape[0] / 8), 3])
        self.display_magnitude_phase(resized_box_blurred_img, label='resized_box')
        cv2.imwrite(self.save_path + 'img2_resized_box_x8.png', resized_box_blurred_img)


img_path = "E:/Computer vision/HW2/EX2/images/"
save_path = "E:/Computer vision/HW2/EX2/results/1/"

part_a = ImageProcessor(img_path, save_path)
part_a.load_image()
part_a.process_resize_700x700()
part_a.process_rotate_resize()
part_a.process_resize_eighth()
part_a.process_gaussian_resize_eighth()
part_a.process_box_blur_resize_eighth()
