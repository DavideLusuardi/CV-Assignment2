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


cap = cv2.VideoCapture(f"{root_dir}/material/CV_basket.mp4")
ret, frame = cap.read()

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(frame.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)
mask = mask[y:y+h, x:x+w]
frame_cut = frame[y:y+h, x:x+w]

box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
print(box)
print(type(box))
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.TrackerKCF_create()
tracker.init(frame, box)

# skip = random.randint(0, 500)
for i in range(1000):
# Capture frame-by-frame
    ret, frame = cap.read()
    # if i < skip:
    #     continue

    frame_cut = frame[y:y+h, x:x+w]
    frame_masked = frame_cut
    #color conversion
    # frame_gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    # frame_masked = cv2.bitwise_and(frame_cut, frame_cut, mask=mask) # apply mask
    # frame = imutils.resize(frame, width=500)
	# (H, W) = frame.shape[:2]

    # grab the new bounding box coordinates of the object
    (success, box) = tracker.update(frame)

    # check to see if the tracking was a success
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
    else:
        print('no success')


    cv2.imshow("Frame", frame)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue

