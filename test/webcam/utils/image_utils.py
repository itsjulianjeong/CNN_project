import cv2
import numpy as np

def resize_image(img, size=(224, 224)):
    return cv2.resize(img, size)

def normalize_image(img):
    img=img.astype("float32")/255.0
    return img  # 3채널 그대로 유지 (H, W, 3)