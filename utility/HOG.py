# import the necessary packages
import numpy as np
import cv2

def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    return x_min, y_min, x_max-x_min, y_max-y_min

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../material/CV_basket.mp4")

background = cv2.imread('../material/background.png', cv2.IMREAD_GRAYSCALE)

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(background.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)

mask = mask[y:y+h, x:x+w]
print(w, h) # 1220 287

# shape = (1080, 810)
# mask = cv2.resize(mask, shape)


# the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640,480))

i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # i += 1
    # if i % 4 != 0:
    #     continue

    frame = frame[y:y+h, x:x+w]
    # resizing for faster detection
    # frame = cv2.resize(frame, shape)
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    frame = cv2.bitwise_and(frame, frame, mask=mask)


    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    # out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)