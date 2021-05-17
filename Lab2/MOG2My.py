import numpy as np
import cv2

learningRate = 0.8 # with -1 openCV tries to decide the best value of alpha
history = 2000 # number of frames to fit the Gaussians
nMixGaussian = 100 # number of Gaussians
backgroundRatio = 0.5
noiseSigma = 1 # noise of the different colors

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history, nMixGaussian, backgroundRatio, noiseSigma)
fgbg = cv2.createBackgroundSubtractorMOG2()

waiting_time = 1
cap = cv2.VideoCapture(0)

for i in range(3000):
    # capture frame by frame
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame, learningRate)
    if i > 20:
        bg = fgbg.getBackgroundImage()
        cv2.imshow('background', bg)

    # display
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    cv2.waitKey(waiting_time)

cap.release()
cv2.destroyAllWindows()