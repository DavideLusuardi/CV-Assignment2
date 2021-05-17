import numpy as np
import cv2

def bs():
    cap = cv2.VideoCapture("../material/CV_basket.mp4") 

    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        fgmask = fgbg.apply(frame)

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgmask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

bs()