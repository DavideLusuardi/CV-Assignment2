import cv2
import numpy
import math

cap = cv2.VideoCapture("../material/Video.mp4")
frames = []

for i in range(1000):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(frame)

cap.release()

def difference(frames, t, N):
    # return cv2.absdiff(frames[t], frames[t+N])
    return math.sqrt((frames[t] - frames[t+N])**2)

# old_frame = frames[0]
# for new_frame in frames[1:]:    
#     res_frame = cv2.absdiff(old_frame, new_frame)
#     old_frame = new_frame
#     cv2.imshow("frame", res_frame)
#     cv2.waitKey(2)

N = 10
for i in range(len(frames)-N):
    res_frame = difference(frames, i, N)
    cv2.imshow("frame", res_frame)
    cv2.waitKey(10)