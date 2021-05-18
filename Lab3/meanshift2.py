import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def update_hist(roi_hist, hsv_roi):
    mask = cv.inRange(hsv_roi, lower_bound, upper_bound)
    mask2 = cv.inRange(hsv_roi, lower_bound2, upper_bound2)
    mask = cv.bitwise_or(mask, mask2)
    cv.imshow('mask', mask)
    new_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv.normalize(new_hist,new_hist,0,255,cv.NORM_MINMAX)
    
    alpha = 0.3
    beta = (1.0 - alpha)
    hist = cv.addWeighted(roi_hist, alpha, new_hist, beta, 0.0)
    return hist


cap = cv.VideoCapture("../material/CV_basket.mp4")
# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 1026, 612, 30, 28 # simply hardcoded the values
x, y, w, h = cv.selectROI('img2', frame, showCrosshair=False)
track_window = (x, y, w, h)

# lower_bound = np.array((0., 60.,32.))
# upper_bound = np.array((180.,255.,255.))
lower_bound = np.array((107., 81., 117.))
upper_bound = np.array((120., 150., 246.))
lower_bound2 = np.array((0., 0., 0.))
upper_bound2 = np.array((180., 255., 130.))

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, lower_bound, upper_bound)
mask2 = cv.inRange(hsv_roi, lower_bound2, upper_bound2)
mask = cv.bitwise_or(mask, mask2)
# roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180]) # TODO: provare con altri channels
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        
        # TODO: update histogram
        roi_hist = update_hist(roi_hist, hsv[y:y+h, x:x+w])
        
        # Draw it on image
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        cv.imshow('img2',img2)
        # cv.imshow('hsv', hsv)
        k = cv.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('p'):
            plt.plot(roi_hist)
            plt.show()
    else:
        break
