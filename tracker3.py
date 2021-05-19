import numpy as np
import cv2
import random, sys

root_dir = '.'
sys.path.insert(1, f'{root_dir}/utility')
import morphological_op as dl
import yolo


def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    # w, h = 1220, 287
    return x_min, y_min, x_max-x_min, y_max-y_min


PERSON_CLASS_ID = 0
def filter_objects(objects, frame, mask_box):
    boxes = list()
    remaining = list()
    for i, object in enumerate(objects):
        box, classID, label, confidence = object
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])

        # print(f"w {w}, h {h}, label {label}")

        if classID == PERSON_CLASS_ID and \
                w < 60 and h < 150 and \
                x > mask_box[0] and x < mask_box[0]+mask_box[2] and y > mask_box[1] and y < mask_box[1]+mask_box[3]:

            # roi = frame[y:y+h, x:x+w]
            # hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # for k, (lb, ub) in bounds.items():
            #     mask = cv2.inRange(hsv_roi, lb, ub)
            #     whitePixels = cv2.countNonZero(mask)
            #     print(f"person_{i}: {whitePixels} {k}")

            boxes.append((box, classID, f'{i}', confidence))
        else:
            remaining.append(object)

    return boxes, remaining

def count_people(people, x_center):
    left_counter = 0
    for person in people:
        box, classID, label, confidence = person
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])

        if x+w/2 < x_center:
            left_counter += 1

    return left_counter, len(people)-left_counter




args = {'yolo':f'{root_dir}/utility/yolo', 'confidence':0.25, 'threshold':0.1}
net, LABELS, COLORS = yolo.load_data(args)
RED = (0,0,255)
mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
dilatation_size = 10
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
# HSV lower and upper bounds
referee_lb = (107, 81, 117)
referee_ub = (120, 150, 246)
blackp_lb = (0, 0, 0)
blackp_ub = (180, 255, 230)
whitep_lb = (126, 0, 176)
whitep_ub = (169, 57, 255)
bounds = {'referee':(referee_lb, referee_ub), 'black_player':(blackp_lb, blackp_ub), 'white_player':(whitep_lb, whitep_ub)}

background = cv2.imread(f'{root_dir}/material/background.png', cv2.IMREAD_GRAYSCALE)
x_center = background.shape[1]/2

x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(background.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)
mask = mask[y:y+h, x:x+w]

background = background[y:y+h, x:x+w]
background = cv2.bitwise_and(background, background, mask=mask)

cap = cv2.VideoCapture(f"{root_dir}/material/CV_basket.mp4")

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
    
    objects = yolo.detect(args, frame, net, LABELS)
    objects, discards = filter_objects(objects, frame, (x,y,w,h))
    print(f"persons: {len(objects)}")
    frame_masked = yolo.draw_labels(frame, objects, [RED])
    frame_masked = yolo.draw_labels(frame, discards, COLORS)
    # frame_masked = cv2.bitwise_and(frame_masked[y:y+h, x:x+w], frame_masked[y:y+h, x:x+w], mask=mask)

    left, right = count_people(objects, x_center)
    print(f"left: {left}\nright: {right}")

    # Show keypoints
    # cv2.imshow('Blobs',blobs)
    # Display the resulting frame
    # cv2.imshow('frame',frame_gray)
    cv2.imshow('frame masked',frame_masked)
    # cv2.imshow('discards',discards_frame)
    cv2.imshow('motion_mask',motion_mask)
    # cv2.imshow('Background',background)
    
    k = cv2.waitKey(100)
    if k == ord('q'): #close video if q is pressed
        break
    elif k == ord('s'):
        # frame = cv2.bitwise_and(frame, frame, mask=motion_mask)
        cv2.imwrite(f'{root_dir}/material/frames/image-{i}.png', frame)
        print(f'image saved: {root_dir}/material/frames/image-{i}.png')
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue