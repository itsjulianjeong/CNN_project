# 화면 출력 함수
# cv2.putText 등

import cv2

# 눈 상태 예측 결과 텍스트로 시각화
# frame: 원본 프레임
# position: 텍스트 표시 위치 (x, y)
# pred: 예측 확률값 (0~1 사이 값)
# threshold: open, close 판단 기준값 (0.5)
def draw_prediction(frame, position, pred, threshold=0.5):
    state="CLOSED" if pred < threshold else "OPEN"
    color=(0, 0, 255) if state == "CLOSED" else (0, 255, 0)
    label=f"{state}: {pred:.2f}"
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)