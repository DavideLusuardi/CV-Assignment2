import numpy as np
import cv2
import random, sys
sys.path.insert(1, '../utility')
import morphological_op as dl
import yolo

def bg_update(frame_gray,bg):
    # alfa = 0.05
    alfa = 0
    bg = np.uint8(bg*(1-alfa) + alfa*frame_gray)
    #bg = frame_gray
    return bg


def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    # w, h = 1220, 287
    return x_min, y_min, x_max-x_min, y_max-y_min

args = {'yolo':'../utility/yolo', 'confidence':0.3, 'threshold':0.1}
net, LABELS, COLORS = yolo.load_data(args)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture("../material/CV_basket.mp4")

background = cv2.imread('../material/background.png', cv2.IMREAD_GRAYSCALE)

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(background.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)
mask = mask[y:y+h, x:x+w]

background = background[y:y+h, x:x+w]
background = cv2.bitwise_and(background, background, mask=mask)

# dl.init()

blank = np.zeros((1,1))
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# params.minDistBetweenBlobs = 20;

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;
    
# Filter by Area.
# params.filterByArea = True
# params.minArea = 20
    
# # Filter by Circularity
# params.filterByCircularity = False
# params.minCircularity = 0.1
    
# # Filter by Convexity
# params.filterByConvexity = False
# params.minConvexity = 0.5
    
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.5
    
# detector = cv2.SimpleBlobDetector_create(params)

dilatation_size = 10
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

skip = random.randint(0, 500)
for i in range(1000):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if i < skip:
        continue

    frame_cut = frame[y:y+h, x:x+w]
    #color conversion
    frame_gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask) # apply mask

    #bg subtraction
    diff = cv2.absdiff(background, frame_gray)
    #mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)

    # dl.src = motion_mask.copy()[450:750, 0:]
    # dl.dilatation(None)

    #update background
    # background = bg_update(frame_gray,background)
    
    # motion_mask = cv2.erode(motion_mask, None, iterations=1)
    motion_mask = cv2.dilate(motion_mask, element, iterations=1)

    # # Detect blobs.
    # reversemask=255-motion_mask
    # keypoints = detector.detect(reversemask)
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
    # blobs = cv2.drawKeypoints(frame_gray, keypoints, blank, (0,0,255)) # , cv2.DRAW_MATCHES_FLAGS_DEFAULT

    frame_masked = cv2.bitwise_and(frame_cut, frame_cut, mask=motion_mask)


    # boxes, weights = hog.detectMultiScale(frame_masked, winStride=(8,8) )
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     cv2.rectangle(frame_masked, (xA, yA), (xB, yB),
    #                       (255, 255, 255), 2)
    
    frame_masked = yolo.detect(args, frame, net, LABELS, COLORS)

    # Show keypoints
    # cv2.imshow('Blobs',blobs)
    # Display the resulting frame
    # cv2.imshow('frame',frame_gray)
    cv2.imshow('frame masked',frame_masked)
    cv2.imshow('motion_mask',motion_mask)
    # cv2.imshow('Background',background)
    
    k = cv2.waitKey(50)
    if k == ord('q'): #close video is q is pressed
        break
    elif k == ord('s'):
        # frame = cv2.bitwise_and(frame, frame, mask=motion_mask)
        cv2.imwrite(f'../material/frames/image-{i}.png', frame)