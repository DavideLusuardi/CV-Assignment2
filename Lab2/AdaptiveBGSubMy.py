import numpy as np
import cv2

alpha = 0.2
waiting_time = 1
cap = cv2.VideoCapture(0)

for i in range(3000):
    # capture frame by frame
    ret, frame = cap.read()

    # color conversion
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if i == 0:
        # set the first frame as background
        background = frame_gray
    else:
        # background subtraction
        diff = cv2.absdiff(background, frame_gray)

        # mask tresholding
        ret2, motion_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # adaptive background subtraction
        background = np.uint8(alpha*frame_gray + (1-alpha)*background)

        # display
        cv2.imshow('frame', frame)
        cv2.imshow('motion_mask', motion_mask)
        cv2.imshow('background', background)
        cv2.waitKey(waiting_time)


cap.release()
cv2.destroyAllWindows()