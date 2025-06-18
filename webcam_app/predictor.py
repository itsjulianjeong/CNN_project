# 메인 로직 (실시간 감지)
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
import pygame
from collections import deque  # 최근 프레임 예측값 추적용

# utils/*.py 에서 함수들 import
# 이미지 전처리 함수 통합 (grayscale, 이미지 크기 조정, 픽셀값 0~1 사이로 정규화)
from webcam_app.utils.image_utils import to_grayscale, resize_image, normalize_image
from webcam_app.utils.draw_result import draw_prediction

def play_beep(beep_path):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(beep_path)
        pygame.mixer.music.play()

def run_detection():
    # 05번 모델 로드
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH=os.path.join(BASE_DIR, "model", "readjusted_model.keras")
    model=tf.keras.models.load_model(MODEL_PATH)

    # MediaPipe 얼굴/눈 탐지
    # face_mesh: 얼굴에서 468개의 landmark 좌표를 찾아줌 -> 눈,코끝, 턱선, 입술 경계 등 정확한 포인트 기반 탐지
    # -> 눈, 코, 입 위치 추출 -> 그걸 기반으로 crop하고 CNN에 넣음
    MP_FACE=mp.solutions.face_mesh  

    # FaceMesh 객체(얼굴 랜드마크 탐지기) 생성
    FACE_MESH=MP_FACE.FaceMesh(
        static_image_mode=False,  # 비디오 스트림(실시간)에서 연속적으로 프레임 처리(True: 정적 이미지)
        max_num_faces=1,  # max_num_faces=1: 최대 1개의 얼굴만 탐지
        refine_landmarks=True,  # 정밀 홍채 추적 (test_facemesh/test_facemesh_landmarks.ipynb 참고)
        min_detection_confidence=0.5,  # 첫 탐지시 얼굴로 인식되는 최소 신뢰도 -> 얼굴 탐지가 약한 경우 높이기
        #min_tracking_confidence=0.5  # 이후 프레임에서 얼굴 추적 신뢰도 -> 얼굴이 잘 추적되지 않는 경우 높이기
    )

    # MediaPipe 얼굴 landmark idx -> 눈 주변 landmark 번호
    # 홍채 중심 제외(473, 478)
    R_EYE_IDX=list(range(469, 473))
    L_EYE_IDX=list(range(474, 478))

    # 웹캠 연결(웹캠 하나만 연결되어 있으니까 0이 기본)
    capture=cv2.VideoCapture(0)  

    # Sound Effect by <a href="https://pixabay.com/users/freesound_community-46691455/?utm_source=link-attribution&utm_medium=referral&utm_campaign=music&utm_content=6387">freesound_community</a> from <a href="https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=music&utm_content=6387">Pixabay</a>
    WARNING_BEEP=os.path.join(BASE_DIR, "utils", "beep.mp3")
    CAPTURE_DIR="webcam_app/captures"  # cv2.imwrite() -> 졸음 상태 스크린샷
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    CLOSED_FRAME_THRESHOLD=15  # 약 1초 (30fps 기준) -> 연속으로 close 상태가 감지돼야 하는 프레임 수
    closed_frame_count=0  # close 상태가 몇 프레임 동안 지속되었는지 count -> 눈을 뜨면 0으로 리셋
    drowsy_start_time=None  # 졸음 상태 시작 시각

    SMOOTHING_WINDOW_SIZE=7
    right_eye_buffer=deque(maxlen=SMOOTHING_WINDOW_SIZE)
    left_eye_buffer=deque(maxlen=SMOOTHING_WINDOW_SIZE)
    
    while capture.isOpened():  # 웹캠이 열려있는 동안 반복
        TorF, frame = capture.read()  # TorF(True/False): 프레임을 제대로 읽었는지, frame: 실제로 읽어온 영상 프레임 (numpy 배열)
        if not TorF: break # 프레임 읽기 실패 시(웹캠 연결이 끊기거나 에러가 나면) 루프 종료

        frame=cv2.flip(frame, 1)  # 좌우 반전
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력을 요구
        
        results=FACE_MESH.process(frame_rgb)  # 얼굴 랜드마크 추출
        if results.multi_face_landmarks:  # None이면 얼굴 인식 실패 -> 건너뜀
            for face_landmarks in results.multi_face_landmarks:  # 여러 명의 얼굴이 탐지되었을 경우를 대비한 반복문
                                                                # max_num_faces=1이니까 1명만 처리 -> 여러 명도 가능
                
                # OpenCV: (height, width, 채널) -> BGR이니까 3 but 안쓰니까 _
                # MediaPipe: landmark 좌표를 정규화된 0~1 범위로 줌 -> 이걸 실제 좌표로 바꾸려면 곱하기 w/h 필요
                h, w, _=frame.shape  
                # 각 눈 좌표 추출 (왼쪽, 오른쪽)
                right_eye_points=[(int(face_landmarks.landmark[idx].x * w),
                                int(face_landmarks.landmark[idx].y * h)) for idx in R_EYE_IDX]
                left_eye_points=[(int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)) for idx in L_EYE_IDX]
                
                # 눈 좌표로 사각형 crop
                # right 눈 주변 4개 점
                r_x1=min([point[0] for point in right_eye_points])  # 왼쪽 x
                r_x2=max([point[0] for point in right_eye_points])  # 오른쪽 x
                r_y1=min([point[1] for point in right_eye_points])  # 위 y
                r_y2=max([point[1] for point in right_eye_points])  # 아래 y
                # left 눈 주변 4개 점
                l_x1=min([point[0] for point in left_eye_points])  # 왼쪽 x
                l_x2=max([point[0] for point in left_eye_points])  # 오른쪽 x
                l_y1=min([point[1] for point in left_eye_points])  # 위 y
                l_y2=max([point[1] for point in left_eye_points])  # 아래 y
                # -> 4개의 점으로부터 사각형의 bounding box를 구하기
                # (rx1, ry1)~(rx2, ry2) 좌표로 자르면 눈 부분만 잘라낼 수 있음

                eyes={"right": (r_x1, r_x2, r_y1, r_y2), "left": (l_x1, l_x2, l_y1, l_y2)}
                
                for eye, (x1, x2, y1, y2) in eyes.items():
                    eye_img=frame[y1:y2, x1:x2]  # (height, width) 순서 -> (y, x) 순서
                    if eye_img.size == 0: continue  # 눈 영역 crop 결과가 비어 있으면(size==0) -> 스킵 후 for 루프로(다음 eye)
                    gray=to_grayscale(eye_img)  # (y, x) -> (H, W, 1)
                    resized=resize_image(gray)  # (86, 86, 1)
                    normalized=normalize_image(resized)  # 0~1 정규화
                    input_tensor=np.expand_dims(normalized, axis=0)  # 배치 차원 추가 -> (1, 86, 86, 1)
                    # 이제 CNN 입력과 일치하는 구조
                    
                    # 배열 형태로 반환받음(i.g., array([[0.xxxx]]))  -> [0][0]은 첫 번째 배치의 첫 번째 출력값
                    # pred는 확률값(threshold 필요)
                    raw_pred=model.predict(input_tensor, verbose=0)[0][0]
                    pred=1.0-raw_pred  # open 확률로 재해석
                    
                    # 각각의 눈 예측값을 버퍼에 저장
                    if eye=="right":
                        right_eye_buffer.append(pred)
                    else:
                        left_eye_buffer.append(pred)
                
                # 버퍼의 평균값을 사용해 예측값을 안정화
                right_mean=np.mean(right_eye_buffer) if right_eye_buffer else 1.0
                left_mean=np.mean(left_eye_buffer) if left_eye_buffer else 1.0
                
                # 안정화된 예측 결과를 시각화
                draw_prediction(frame, (r_x1, r_y1-10), right_mean, box_coords=(r_x1, r_y1, r_x2, r_y2))
                draw_prediction(frame, (l_x1, l_y1-10), left_mean, box_coords=(l_x1, l_y1, l_x2, l_y2))
                
                CLOSE_THRESHOLD=0.7
                if all(p < CLOSE_THRESHOLD for p in [right_mean, left_mean]):
                    closed_frame_count += 1  # 감김 프레임 누적 시키고
                    if closed_frame_count >= CLOSED_FRAME_THRESHOLD:  # 1초 이상 close 지속됐을때
                        current_time=time.time()
                        
                        if drowsy_start_time is None:
                            drowsy_start_time=current_time
                            timestamp=int(current_time)
                            capture_path=os.path.join(CAPTURE_DIR, f"drowsy_{timestamp}.jpg")
                            cv2.imwrite(capture_path, frame)
                            play_beep(WARNING_BEEP)
                        elif current_time - drowsy_start_time >= 1.0:
                            drowsy_start_time=current_time
                            timestamp=int(current_time)
                            capture_path=os.path.join(CAPTURE_DIR, f"drowsy_{timestamp}.jpg")
                            cv2.imwrite(capture_path, frame)
                            play_beep(WARNING_BEEP)
                            
                        # 빨간색 오버레이 적용
                        overlay=np.full(frame.shape, (0, 0, 255), dtype=np.uint8)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                else:
                    closed_frame_count=0
                    drowsy_start_time=None

        cv2.imshow("Eye State Detector", frame)  # 이미지 창 열기
        
        # esc누르면 종료
        # cv2.waitKey(1): 1ms 동안 키보드 입력 대기(1 프레임 단위)
        # & 0xFF: Windows와 Linux 환경에서 키보드 입력을 일관되게 처리하기 위한 마스크
        # 27: sec
        if cv2.waitKey(1) & 0xFF == 27: break  

    capture.release()  # 웹캠 자원 반납
    cv2.destroyAllWindows()  # 이미지 창 닫기
    
if __name__ == "__main__":
    run_detection()