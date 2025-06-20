import cv2
import numpy as np

IMG_SIZE=224

def preprocess_eye(image, eye_landmarks):
    x_coords=[pt[0] for pt in eye_landmarks]
    y_coords=[pt[1] for pt in eye_landmarks]

    x_min, x_max=min(x_coords), max(x_coords)
    y_min, y_max=min(y_coords), max(y_coords)

    w=x_max-x_min
    h=y_max-y_min

    margin_x=int(w*0.4)
    margin_y=int(h*0.4)

    x1=max(x_min-margin_x, 0)
    y1=max(y_min-margin_y, 0)
    x2=min(x_max+margin_x, image.shape[1])
    y2=min(y_max+margin_y, image.shape[0])

    eye=image[y1:y2, x1:x2]
    gray=cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized=resized.astype("float32")/255.0
    return np.expand_dims(normalized, axis=-1)

def preprocess_eye_image(eye_img, target_size=224):
    """
    - eye_img: numpy array (RGB)
    - return: grayscale -> resize -> normalize (0~1)
    """
    gray=cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
    resized=cv2.resize(gray, (target_size, target_size))
    norm_img=resized.astype(np.float32)/255.0
    norm_img=np.expand_dims(norm_img, axis=-1)  # 채널 차원 추가
    return norm_img