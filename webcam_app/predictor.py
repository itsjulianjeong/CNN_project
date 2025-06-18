import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
import pygame
from collections import deque

from webcam_app.utils.image_utils import to_grayscale, resize_image, normalize_image
from webcam_app.utils.draw_result import draw_prediction
from webcam_app.utils.sound_utils import play_beep

def run_detection():
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH=os.path.join(BASE_DIR, "model", "model_final.keras")
    model=tf.keras.models.load_model(MODEL_PATH)

    MP_FACE=mp.solutions.face_mesh
    FACE_MESH=MP_FACE.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # 눈 주변 landmark index (홍채 중심 제외)
    R_EYE_IDX=list(range(469, 473))
    L_EYE_IDX=list(range(474, 478))

    capture=cv2.VideoCapture(0)

    # 경고음
    WARNING_BEEP=os.path.join(BASE_DIR, "utils", "beep.mp3")
    CAPTURE_DIR="webcam_app/captures"
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    CLOSED_FRAME_THRESHOLD=15  # 0.5초 지속 기준
    closed_frame_count=0
    drowsy_start_time=None

    SMOOTHING_WINDOW_SIZE=7
    right_eye_buffer=deque(maxlen=SMOOTHING_WINDOW_SIZE)
    left_eye_buffer=deque(maxlen=SMOOTHING_WINDOW_SIZE)
    
    while capture.isOpened():
        TorF, frame=capture.read()
        if not TorF: break

        frame=cv2.flip(frame, 1)
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=FACE_MESH.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _=frame.shape

                right_eye_points=[(int(face_landmarks.landmark[idx].x * w),
                                   int(face_landmarks.landmark[idx].y * h)) for idx in R_EYE_IDX]
                left_eye_points=[(int(face_landmarks.landmark[idx].x * w),
                                  int(face_landmarks.landmark[idx].y * h)) for idx in L_EYE_IDX]

                r_x1, r_x2 = min(p[0] for p in right_eye_points), max(p[0] for p in right_eye_points)
                r_y1, r_y2 = min(p[1] for p in right_eye_points), max(p[1] for p in right_eye_points)
                l_x1, l_x2 = min(p[0] for p in left_eye_points), max(p[0] for p in left_eye_points)
                l_y1, l_y2 = min(p[1] for p in left_eye_points), max(p[1] for p in left_eye_points)

                eyes={"right": (r_x1, r_x2, r_y1, r_y2), "left": (l_x1, l_x2, l_y1, l_y2)}
                
                for eye, (x1, x2, y1, y2) in eyes.items():
                    eye_img=frame[y1:y2, x1:x2]
                    if eye_img.size == 0: continue
                    gray=to_grayscale(eye_img)
                    resized=resize_image(gray)
                    normalized=normalize_image(resized)
                    input_tensor=np.expand_dims(normalized, axis=0)
                    raw_pred=model.predict(input_tensor, verbose=0)[0][0]
                    pred=1.0 - raw_pred  # open 확률로 재해석

                    if eye=="right":
                        right_eye_buffer.append(pred)
                    else:
                        left_eye_buffer.append(pred)

                right_mean=np.mean(right_eye_buffer) if right_eye_buffer else 1.0
                left_mean=np.mean(left_eye_buffer) if left_eye_buffer else 1.0

                draw_prediction(frame, (r_x1, r_y1-10), right_mean, box_coords=(r_x1, r_y1, r_x2, r_y2))
                draw_prediction(frame, (l_x1, l_y1-10), left_mean, box_coords=(l_x1, l_y1, l_x2, l_y2))

                CLOSE_THRESHOLD=0.7
                if all(p < CLOSE_THRESHOLD for p in [right_mean, left_mean]):
                    closed_frame_count += 1
                    if closed_frame_count >= CLOSED_FRAME_THRESHOLD:
                        current_time=time.time()
                        if drowsy_start_time is None:
                            drowsy_start_time=current_time
                            timestamp=int(current_time)
                            cv2.imwrite(os.path.join(CAPTURE_DIR, f"drowsy_{timestamp}.jpg"), frame)
                            play_beep(WARNING_BEEP)
                        elif current_time - drowsy_start_time >= 1.0:
                            drowsy_start_time=current_time
                            timestamp=int(current_time)
                            cv2.imwrite(os.path.join(CAPTURE_DIR, f"drowsy_{timestamp}.jpg"), frame)
                            play_beep(WARNING_BEEP)
                        overlay=np.full(frame.shape, (0, 0, 255), dtype=np.uint8)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                else:
                    closed_frame_count=0
                    drowsy_start_time=None

        cv2.imshow("Eye State Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()