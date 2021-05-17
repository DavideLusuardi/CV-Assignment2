import cv2
import numpy as np

cap = cv2.VideoCapture("../material/CV_basket.mp4")

last_pos = None
ret,frame = cap.read()
result = frame.copy()
points = list()

def mousemove(event, x,y,s,p):
    global points, frame, last_pos, result

    last_pos = (x,y)

    if event == cv2.EVENT_LBUTTONUP:
        print(x,y)
        points.append((x,y))

        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1],(0,0,0),thickness=4)
            result = frame.copy()

    elif len(points) > 0:
        result = frame.copy()
        cv2.line(result, points[-1], (x,y),(0,0,0),thickness=4)



cv2.namedWindow("Click")
cv2.setMouseCallback("Click", mousemove)

while(True):
    cv2.imshow("Click", result)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print(points)
        break

cv2.destroyAllWindows()
