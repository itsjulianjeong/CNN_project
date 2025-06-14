# 이미지 전처리 함수 통합 (grayscale, 이미지 크기 조정, 픽셀값 0~1 사이로 정규화)

import cv2
import numpy as np

def to_grayscale(image):
    # BGR -> 흑백 1채널로 변환
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # (H, W, 1)

def resize_image(image, size=(86, 86)):
    # 모델 입력 크기로 리사이즈 86x86
    return cv2.resize(image, size).reshape(size[1], size[0], 1)  # (H, W, 1)

def normalize_image(image):
    # 픽셀 값을 0~1 범위로 정규화
    return image.astype("float32") / 255.0
