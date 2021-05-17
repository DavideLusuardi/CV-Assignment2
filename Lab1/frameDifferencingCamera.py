import cv2
import numpy
import math


def difference(frames, t, N):
    # return cv2.absdiff(frames[t], frames[t+N])
    return math.sqrt((frames[t] - frames[t+N])**2)


cap = cv2.VideoCapture(0)
frames = []

N = 4
thr = 50
for i in range(200):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(frame)
    
    if i>=N:
        diff = cv2.absdiff(frames[-N-1], frames[-1])

        ret2, motion_mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

        cv2.imshow("frame", motion_mask)
        cv2.waitKey(1)

cap.release()
