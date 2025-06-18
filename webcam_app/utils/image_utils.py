import cv2
import numpy as np

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # (H, W, 1)

def resize_image(image, size=(86, 86)):
    return cv2.resize(image, size).reshape(size[1], size[0], 1)  # (H, W, 1)

def normalize_image(image):
    return image.astype("float32") / 255.0
