import cv2

def draw_prediction(frame, coord, pred_prob, box_coords=None):
    label=f"{pred_prob:.2f}"
    color=(0, 0, 255) if pred_prob < 0.7 else (0, 255, 0)
    cv2.putText(frame, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if box_coords:
        x1, y1, x2, y2=box_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
