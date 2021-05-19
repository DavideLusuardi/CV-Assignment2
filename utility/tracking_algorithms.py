import numpy as np
import cv2
import random, sys

root_dir = '..'

def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    # w, h = 1220, 287
    return x_min, y_min, x_max-x_min, y_max-y_min


dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
def background_subtraction(image, background):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #bg subtraction
    diff = cv2.absdiff(background, frame_gray)
    #mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, element, iterations=1)
    frame_masked = cv2.bitwise_and(image, image, mask=motion_mask)

    return frame_masked, motion_mask


cap = cv2.VideoCapture(f"{root_dir}/material/CV_basket.mp4")
skip = random.randint(1, 500)
skip = 1
for i in range(skip):
    ret, frame = cap.read()

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(frame.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)
mask = mask[y:y+h, x:x+w]
frame_cut = frame[y:y+h, x:x+w]

background = cv2.imread(f'{root_dir}/material/background.png', cv2.IMREAD_GRAYSCALE)
background = background[y:y+h, x:x+w]

frame_masked, motion_mask = background_subtraction(frame_cut, background)


lower_bound = np.array((107., 81., 117.))
upper_bound = np.array((120., 150., 246.))

hsv_roi =  cv2.cvtColor(frame_cut, cv2.COLOR_BGR2HSV)
referee_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
referee_mask = cv2.dilate(referee_mask, element, iterations=2)

frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=referee_mask)

box = cv2.selectROI("Frame", frame_masked, fromCenter=False, showCrosshair=False)
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.TrackerKCF_create()
tracker.init(frame_masked, box)


kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1


while True:
# Capture frame-by-frame
    ret, frame = cap.read()

    frame_cut = frame[y:y+h, x:x+w]
    frame_masked, motion_mask = background_subtraction(frame_cut, background)

    hsv_roi =  cv2.cvtColor(frame_cut, cv2.COLOR_BGR2HSV)
    referee_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    referee_mask = cv2.dilate(referee_mask, element, iterations=2)
    # joint_mask = cv2.bitwise_and(referee_mask, motion_mask)

    frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=referee_mask)


    #color conversion
    # frame_gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    # frame_masked = cv2.bitwise_and(frame_cut, frame_cut, mask=mask) # apply mask
    # frame = imutils.resize(frame, width=500)
	# (H, W) = frame.shape[:2]

    # grab the new bounding box coordinates of the object
    (success, box) = tracker.update(frame_masked)

    # check to see if the tracking was a success
    if success:
        (xr, yr, wr, hr) = [int(v) for v in box]
        cv2.rectangle(frame_masked, (xr, yr), (xr + wr, yr + hr),(0, 255, 0), 2)
        # kalman.correct(np.array([[np.float32(xr)],[np.float32(yr)]]))
        # prediction = kalman.predict()
        # cpx, cpy = int(prediction[0].astype(int)),int(prediction[1].astype(int))
        # cv2.circle(frame_masked, (cpx,cpy), radius=3, color=(0, 0, 255), thickness=-1)
    else:
        print('no success')


    cv2.imshow("frame_masked", frame_masked)
    cv2.imshow("motion_mask", motion_mask)
    cv2.imshow("referee_mask", referee_mask)

    k = cv2.waitKey(100)
    if k == ord('q'):
        break
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue

