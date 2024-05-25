import cv2 as cv
import os
import numpy as np
import random

# Specify the folder with the images to be augmented
folder_path = r'D:\python_code\pythonProject\Coursera\XuLyAnh\crop'
save_path = r'D:\python_code\pythonProject\Coursera\XuLyAnh\augument'

# Perform augmentation for each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path)

        # Apply augmentation techniques to create new images
        # Flipping
        flipped_img = cv.flip(img, 1)

        # Gaussian Blur
        gaussian_blur_img = cv.GaussianBlur(img, (5, 5), 0)

        # Random rotation
        angle = random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv.warpAffine(img, M, (w, h))

        # Color shifting
        B_shift = random.randint(-50, 50)
        G_shift = random.randint(-50, 50)
        R_shift = random.randint(-50, 50)
        B, G, R = cv.split(img)
        B = np.clip(B + B_shift, 0, 255).astype(np.uint8)
        G = np.clip(G + G_shift, 0, 255).astype(np.uint8)
        R = np.clip(R + R_shift, 0, 255).astype(np.uint8)
        color_shifted_img = cv.merge([B, G, R])

        # Save new images
        new_filename_flipped = "flipped_" + filename
        new_filename_blurred = "blurred_" + filename
        new_filename_rotated = "rotated_" + filename
        new_filename_color_shifted = "color_shifted_" + filename

        new_img_path_flipped = os.path.join(save_path, new_filename_flipped)
        new_img_path_blurred = os.path.join(save_path, new_filename_blurred)
        new_img_path_rotated = os.path.join(save_path, new_filename_rotated)
        new_img_path_color_shifted = os.path.join(save_path, new_filename_color_shifted)

        cv.imwrite(new_img_path_flipped, flipped_img)
        cv.imwrite(new_img_path_blurred, gaussian_blur_img)
        cv.imwrite(new_img_path_rotated, rotated_img)
        cv.imwrite(new_img_path_color_shifted, color_shifted_img)

print("Image augmentation completed and saved.")