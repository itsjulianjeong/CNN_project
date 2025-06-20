# import sys
# sys.path.append("test")

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque
from webcam.utils.image_utils import preprocess_eye_image
from webcam.utils.draw_result import draw_prediction

# 모델 불러오기
model=tf.keras.models.load_model("result/model.keras")

# MediaPipe 초기화
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5)

# 눈 좌표 인덱스 (왼쪽, 오른쪽)
LEFT_EYE_IDX=[33, 133]
RIGHT_EYE_IDX=[362, 263]

# 설정
CLOSED_FRAME_THRESHOLD=15
FRAME_HISTORY=deque(maxlen=30)

# 전처리 및 예측 함수
def predict_eye_state(eye_img):
    input_tensor=np.expand_dims(eye_img, axis=0)
    pred=model.predict(input_tensor, verbose=0)[0][0]
    return "closed" if pred<0.5 else "open"

# 실행 함수
def run_detection():
    cap=cv2.VideoCapture(0)
    closed_counter=0

    while cap.isOpened():
        ret, frame=cap.read()
        if not ret:
            break

        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result=face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            h, w=frame.shape[:2]
            landmarks=result.multi_face_landmarks[0].landmark

            # 눈 영역 추출
            left_eye_pts=[landmarks[i] for i in LEFT_EYE_IDX]
            right_eye_pts=[landmarks[i] for i in RIGHT_EYE_IDX]

            lx1, ly1=int(left_eye_pts[0].x*w), int(left_eye_pts[0].y*h)
            lx2, ly2=int(left_eye_pts[1].x*w), int(left_eye_pts[1].y*h)
            rx1, ry1=int(right_eye_pts[0].x*w), int(right_eye_pts[0].y*h)
            rx2, ry2=int(right_eye_pts[1].x*w), int(right_eye_pts[1].y*h)

            # 박스 범위 확장
            margin=10
            left_eye=frame[max(0, min(ly1, ly2)-margin):min(h, max(ly1, ly2)+margin),
                           max(0, min(lx1, lx2)-margin):min(w, max(lx1, lx2)+margin)]
            right_eye=frame[max(0, min(ry1, ry2)-margin):min(h, max(ry1, ry2)+margin),
                            max(0, min(rx1, rx2)-margin):min(w, max(rx1, rx2)+margin)]

            # 둘 다 존재할 때만 예측
            if left_eye.size and right_eye.size:
                left_proc=preprocess_eye_image(left_eye)
                right_proc=preprocess_eye_image(right_eye)

                left_state=predict_eye_state(left_proc)
                right_state=predict_eye_state(right_proc)

                state="closed" if left_state=="closed" and right_state=="closed" else "open"

                if state=="closed":
                    closed_counter+=1
                else:
                    closed_counter=0

                FRAME_HISTORY.append(state)
                frame=draw_prediction(frame, state, closed_counter)

                if closed_counter>=CLOSED_FRAME_THRESHOLD:
                    overlay=np.full_like(frame, (0, 0, 255), dtype=np.uint8)
                    frame=cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1)&0xFF==27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run_detection()