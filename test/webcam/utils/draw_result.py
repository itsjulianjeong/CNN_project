import cv2

def draw_prediction(frame, state, closed_counter, threshold=15):
    h, w=frame.shape[:2]

    if state=="closed":
        text="DROWSY"
        color=(0, 0, 255)
    else:
        text="AWAKE"
        color=(0, 255, 0)

    cv2.putText(frame, f"State: {text}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Closed Frames: {closed_counter}/{threshold}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame
