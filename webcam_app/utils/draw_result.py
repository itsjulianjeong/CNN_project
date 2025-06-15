# 화면 출력 함수
# cv2.putText 등

import cv2

# 눈 상태 예측 결과 텍스트로 시각화
# frame: 원본 프레임
# position: 텍스트 표시 위치 (x, y)
# pred: 예측 확률값 (0~1 사이 값)
def draw_prediction(frame, position, pred, box_coords=None):
    green = int(pred * 255)
    red = 255 - green
    color = (0, green, red)  # BGR

    # 텍스트: 확률값만 (겹치지 않게 줄임)
    cv2.putText(frame, f"{pred:.2f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    # 박스 그리기
    if box_coords:
        x1, y1, x2, y2=box_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)