import cv2
import numpy as np
import math
import torch
import sys

DEBUG = False
DELAY = 100 if DEBUG else 1

FPS = 25

RED = (0,0,255)
BLUE = (255,0,0)
ORANGE = (0,165,255)
VIOLET = (225,49,139)
PURPLE = (245,71,201)
WHITE = (255,255,255)
BROWN = (33,67,101)

WIDTH = 32
HEIGHT = 64

RIGHT = 0
LEFT = 1
players_location = None
last_possession_change = None


def get_motion_mask(frame, background_gray, mask, dilatation_kernel):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask) # apply mask

    # bg subtraction
    diff = cv2.absdiff(background_gray, frame_gray)

    # mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, dilatation_kernel, iterations=1)
    return motion_mask


def get_mask_from_points(points, dim):
    points = points.reshape((-1,1,2))
    mask = np.zeros(dim, np.uint8)
    cv2.fillPoly(mask,[points],255)
    return mask


def filter_predictions(predictions, mask):
    valid_predictions = list()
    bad_predictions = list()
    for image_predictions in predictions:
        for xmin, ymin, xmax, ymax, confidence, cl in image_predictions:
            xmin, ymin, xmax, ymax, confidence, cl = int(xmin.cpu()), int(ymin.cpu()), int(xmax.cpu()), int(ymax.cpu()), float(confidence.cpu()), float(cl.cpu())
            if mask[(ymin+ymax)//2][(xmin+xmax)//2] == 255:
                valid_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))
            else:
                bad_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))
    
    return valid_predictions, bad_predictions


# Euclidean distance
def l2norm(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


target_hist = None
last_detection = None
def track(frame, motion_mask, track_window, predictions, kalman):
    global target_hist, last_detection

    (xw, yw, ww, hw) = track_window
    track_center = (xw+ww//2, yw+hw//2)
    
    best_prediction = [float('inf'), None, None] # distance, histogram, prediction window
    for i, (x1p, y1p, x2p, y2p, confidence, cl) in enumerate(predictions):
        prediction_center = ((x2p+x1p)/2, (y2p+y1p)/2)
        euclidean_distance = l2norm(track_center, prediction_center)
        if euclidean_distance > 50:
            if DEBUG: print(f'skip due to distance {euclidean_distance}')
            continue
        
        img = frame[y1p:y2p, x1p:x2p]
        m = motion_mask[y1p:y2p, x1p:x2p]
        hist = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        hist_distance = 0
        if target_hist is not None:
            hist_distance = cv2.compareHist(hist, target_hist, cv2.HISTCMP_BHATTACHARYYA)

        if hist_distance > 0.3:
            if DEBUG: print('skip due to histogram')
            continue

        distance = euclidean_distance + hist_distance*100
        if DEBUG: print(f"prediction {i}:\tdistance {euclidean_distance},\thist_distance {hist_distance},\tweighted_sum {distance}")
        if best_prediction[0] > distance:
            best_prediction[0] = distance
            best_prediction[1] = hist
            best_prediction[2] = (x1p, y1p, x2p-x1p, y2p-y1p)

    pred_distance, pred_hist, pred_window = best_prediction
    if pred_distance < 100:
        track_window = pred_window
        centerx, centery = pred_window[0]+pred_window[2]/2, pred_window[1]+pred_window[3]/2
        kalman.correct(np.array([[np.float32(centerx)],[np.float32(centery)]]))
        kalman.predict()
        target_hist = pred_hist
        last_detection = track_window
        predicted = False
    else:        
        current_pre = kalman.predict()
        cpx, cpy = current_pre[0].astype(int).item(), current_pre[1].astype(int).item()
        track_window = (cpx-WIDTH//2, cpy-HEIGHT//2, WIDTH, HEIGHT) # TODO: get last average width and height and check not going out of borders
        predicted = True
    
    if DEBUG:
        print(track_window)
        print(f"predicted {predicted}")
        print('-----------------------------------------------------------------------')

    return track_window, predicted


def draw_trajectory(frame, tracking_points):
    px, py = tracking_points[0][:2]
    for i in range(len(tracking_points)):
        cx, cy, predicted = tracking_points[i]
        
        if not predicted:
            cv2.line(frame, (px, py),(cx, cy), BROWN, 2)
        else:
            cv2.line(frame, (px, py),(cx, cy), PURPLE, 3)

        px, py = cx, cy
    
    return frame

background_gray = cv2.imread(f'../material/background.png', cv2.IMREAD_GRAYSCALE)

mask_points1 = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
mask1 = get_mask_from_points(mask_points1, dim=background_gray.shape[:2])

background_gray = cv2.bitwise_and(background_gray, background_gray, mask=mask1)

dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

mask_points2 = np.array([(56, 641), (323, 548), (511, 557), (696, 561), (901, 554), (1063, 543), (1353, 630), (1089, 700), (708, 739), (368, 715), (89, 661), (49, 644)], np.int32)
mask2 = get_mask_from_points(mask_points2, dim=background_gray.shape[:2])

# detector
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
detector.classes = [0] # find only object of class person


kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.3
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1



cap = cv2.VideoCapture('../material/CV_basket.mp4')
ret, frame = cap.read()
if ret == False:
    sys.exit(1)

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (1422,1080))

track_window = cv2.selectROI('frame', frame, showCrosshair=False)
cx, cy, cw, ch = track_window
tracking_points = [(cx+cw//2, cy+ch//2, False)]

avg_right, avg_left = 5, 5
alpha = 0.8
frame_index = 0
last_detection_frame = None
while cap.isOpened() and ret:    
    # import random
    # if random.randint(0,10) != 0:
    #     ret, frame = cap.read()
    #     continue

    motion_mask = get_motion_mask(frame, background_gray, mask1, dilatation_kernel=element)

    with torch.no_grad():
        frame = frame[:,:,::-1]
        results = detector(frame)
        # for iiimg in results.render():
        #     cv2.imshow('render', iiimg)
        frame = frame[:,:,::-1]

    valid_predictions, bad_predictions = filter_predictions(results.xyxy, mask2) # TODO: try change mask

    if track_window is not None: # check if tracking should continue
        track_window, predicted = track(frame, motion_mask, track_window, valid_predictions+bad_predictions, kalman) # TODO: try with only valid_predictions
        
        cx, cy, cw, ch = track_window
        tracking_points.append((cx+cw//2, cy+ch//2, predicted))        

        if not predicted:
            last_detection_frame = frame_index
        
        if predicted and last_detection is not None and (l2norm(last_detection[:2], track_window[:2]) > 200 or frame_index-last_detection_frame > 10):
            print('target missed!!!!')
            track_window = None

    frame = draw_trajectory(frame, tracking_points)

    left_counter = 0
    right_counter = 0
    for xmin, ymin, xmax, ymax, confidence, cl in valid_predictions:
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 2)
        if (xmax+xmin)/2 < frame.shape[1]/2:
            left_counter += 1
        else:
            right_counter += 1
            
    avg_left = alpha*avg_left + (1-alpha)*left_counter
    avg_right = alpha*avg_right + (1-alpha)*right_counter
    # print(f"avg_left {avg_left}")
    # print(f"avg right {avg_right}")

    for xmin, ymin, xmax, ymax, confidence, cl in bad_predictions:
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BLUE, 2)

    if track_window is not None:
        (xw, yw, ww, hw) = track_window
        if not predicted:
            cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), BROWN, 3)
        else:
            cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), PURPLE, 3)
    
    cv2.putText(frame, f'people: {len(valid_predictions)+len(bad_predictions)} (total), {len(valid_predictions)} (court)', 
        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1, cv2.LINE_AA)
    if last_possession_change is not None and frame_index - last_possession_change < FPS*3:
        cv2.putText(frame, f'possession change to {"right" if players_location==RIGHT else "left"}',
            (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    out.write(frame)

    if players_location != LEFT and avg_right < 0.1: #right_counter == 0 and 
        players_location = LEFT
        last_possession_change = frame_index
        print('change possession to left')

    if players_location != RIGHT and avg_left < 0.1: # left_counter == 0 and 
        players_location = RIGHT
        last_possession_change = frame_index
        print('change possession to right')


    k = cv2.waitKey(DELAY)
    if k == ord('q'): # quit
        break
    elif k == ord(' '): # pause
        while cv2.waitKey(0) != ord(' '):
            continue

    ret, frame = cap.read()
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()