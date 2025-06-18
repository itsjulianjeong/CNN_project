import cv2

def draw_prediction(frame, position, pred, box_coords=None):
    green=int(pred * 255)
    red=255 - green
    color=(0, green, red)  # BGR

    cv2.putText(frame, f"{pred:.2f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # 박스 그리기
    if box_coords:
        x1, y1, x2, y2=box_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)