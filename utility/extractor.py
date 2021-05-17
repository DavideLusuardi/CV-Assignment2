import cv2
import numpy as np
import os, time

# TODO: non funzionante

cap = cv2.VideoCapture("../material/CV_basket.mp4")

frames = list()
while cap.isOpened():
    #read video
    ret, frame = cap.read()
    if ret == False:
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(50) and (0xFF == ord('q') or 0xFF == ord('a')):
        print('enter')
        if 0xFF == ord('q'): #close video is q is pressed
            print('break')
            break
        elif 0xFF == ord('a'):
            print('append')
            frames.append(frame.copy())

if len(frames) > 0:
    path = '../material/frames/%d' % time.time()
    os.mkdir(path)

for i, frame in enumerate(frames):
    cv2.imwrite('%s-%d' % ('../material/frames/%d'%time.time(), i+1), frame)

cap.release()
cv2.destroyAllWindows()