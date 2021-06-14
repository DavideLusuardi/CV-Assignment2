import cv2
import numpy as np
import math
import torch
import sys

DEBUG = False
DELAY = 100 if DEBUG else 1

# video fps and resolution
FPS = 25
RESOLUTION = (1422,1080)

# some colors
RED = (0,0,255)
BLUE = (255,0,0)
PURPLE = (245,71,201)
WHITE = (255,255,255)
BROWN = (33,67,101)

# width and height of the kalman predicted window
WIDTH = 32
HEIGHT = 64

# stop tracking when max number of frames or max distance 
# without detecting the player have been reached
MAX_FRAMES_NO_DETECTION = 40
MAX_DISTANCE_NO_DETECTION = 200

# players location
RIGHT = 0
LEFT = 1
players_location = None


def get_motion_mask(frame, background_gray, mask, dilatation_kernel):
    ''' Calculate the motion mask applying background subtraction between the frame and the background_gray.
    The background image has been obtained merging respectively the left and the right part
    of two different frames representing the court without people.
    '''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask) # apply mask

    # bg subtraction
    diff = cv2.absdiff(background_gray, frame_gray)

    # mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    # apply a dilatation transform
    motion_mask = cv2.dilate(motion_mask, dilatation_kernel, iterations=1)
    return motion_mask


def get_mask_from_points(points, dim):
    ''' Calculate the mask of the court from a list of points.
    '''
    points = points.reshape((-1,1,2))
    mask = np.zeros(dim, np.uint8)
    cv2.fillPoly(mask,[points],255)
    return mask


def filter_predictions(predictions, court_mask):
    ''' Filter the predictions found by the detector.
    A prediction is considered valid if its center point is within the court.
    '''
    valid_predictions = list()
    bad_predictions = list()
    for image_predictions in predictions:
        for xmin, ymin, xmax, ymax, confidence, cl in image_predictions:
            xmin, ymin, xmax, ymax, confidence, cl = int(xmin.cpu()), int(ymin.cpu()), int(xmax.cpu()), int(ymax.cpu()), float(confidence.cpu()), float(cl.cpu())
            # check if the center of the predicted window is in the court mask (has value 255)
            if court_mask[(ymin+ymax)//2][(xmin+xmax)//2] == 255:
                valid_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))
            else:
                bad_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))
    
    return valid_predictions, bad_predictions


def l2norm(p1, p2):
    ''' Calculate the Euclidean distance between two points.
    '''
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# histogram and window of the last tracked detection
target_hist = None
last_detection = None
def track(frame, motion_mask, track_window, predictions, kalman):
    ''' Perform the tracking of the person. Knowing the previous track_window, the function 
    calculates the new window: consider the best prediction as the one that has the lower weighted
    sum of its Euclidean and histogram distance from the previous track_window. When that sum is
    lower than a dynamic threshold, the prediction is considered valid otherwise it is estimated by
    a Kalman filter.
    '''
    global target_hist, last_detection

    (xw, yw, ww, hw) = track_window
    track_center = (xw+ww//2, yw+hw//2)
    
    best_prediction = [float('inf'), None, None] # distance, histogram, prediction window
    for i, (x1p, y1p, x2p, y2p, confidence, cl) in enumerate(predictions):
        prediction_center = ((x2p+x1p)/2, (y2p+y1p)/2)

        # calculate the Euclidean distance between the track_window center and the prediction center
        euclidean_distance = l2norm(track_center, prediction_center)
        # discard the prediction if the distance is greater than the dynamic threshold calculated 
        # wrt. the last frame in which the track_window was equal to a prediction of Yolo (i.e. it has been 
        # not estimated by the Kalman filter)
        if euclidean_distance > 20+50*np.tanh(frame_index-last_detection_frame-1):
            if DEBUG: print(f'discard due to distance {euclidean_distance}')
            continue
        
        img = frame[y1p:y2p, x1p:x2p]
        m = motion_mask[y1p:y2p, x1p:x2p]
        # calculate the normalized histogram of the prediction considering the motion mask
        hist = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        hist_distance = 0
        if target_hist is not None:
            # calculate the Bhattacharyya distance between the prediction and the target histogram
            hist_distance = cv2.compareHist(hist, target_hist, cv2.HISTCMP_BHATTACHARYYA)

        # discard the prediction if the histogram distance is greater than the following dynamic threshold
        if hist_distance > 0.3+0.2*np.tanh(frame_index-last_detection_frame-1):
            if DEBUG: print(f'discard due to histogram {hist_distance}')
            continue

        # weighted sum of Euclidean and histogram distances
        distance = euclidean_distance + hist_distance*200
        if DEBUG: print(f"prediction {i}:\tdistance {euclidean_distance},\thist_distance {hist_distance},\tweighted_sum {distance}")
        if best_prediction[0] > distance:
            best_prediction[0] = distance
            best_prediction[1] = hist
            best_prediction[2] = (x1p, y1p, x2p-x1p, y2p-y1p)

    pred_distance, pred_hist, pred_window = best_prediction
    # discard the best prediction if the weighted distance is greater than the following dynamic threshold
    if pred_distance < 40+80*np.tanh(frame_index-last_detection_frame-1):
        track_window = pred_window
        centerx, centery = pred_window[0]+pred_window[2]/2, pred_window[1]+pred_window[3]/2
        # correct the Kalman filter
        kalman.correct(np.array([[np.float32(centerx)],[np.float32(centery)]]))
        kalman.predict()
        target_hist = pred_hist
        last_detection = track_window
        predicted = False
    else:
        # use the Kalman filter to predict the track_window
        current_pre = kalman.predict()
        cpx, cpy = current_pre[0].astype(int).item(), current_pre[1].astype(int).item()
        if frame_index > 40 and cpx > 0 and cpx < frame.shape[1] and cpy > 0 and cpy < frame.shape[0]:
            track_window = (cpx-WIDTH//2, cpy-HEIGHT//2, WIDTH, HEIGHT)
        predicted = True
    
    if DEBUG:
        print(track_window)
        print(f"predicted {predicted}")
        print('-----------------------------------------------------------------------')

    # return the new track_window and a boolean flag that indicates if the window is predicted by the Kalman filter
    return track_window, predicted


def plot_trajectory(frame, tracking_points):
    ''' Draw the tracking trajectory from the list of points in tracking_points. '''

    px, py = tracking_points[0][:2]
    for i in range(1, len(tracking_points)):
        cx, cy, predicted = tracking_points[i]
        
        # Draw a brown line when the prediction is taken from the detector and
        # a purple line when it is calculated by the Kalman filter
        if not predicted:
            cv2.line(frame, (px, py),(cx, cy), BROWN, 3)
        else:
            cv2.line(frame, (px, py),(cx, cy), PURPLE, 3)

        px, py = cx, cy
    
    return frame


def put_text(frame, valid_predictions, bad_predictions, players_location, possession_changes):
    ''' Display on the frame the number people detected and ball possession information. '''

    cv2.putText(frame, f'people: {len(valid_predictions)+len(bad_predictions)} (total), {len(valid_predictions)} (court)', 
        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1, cv2.LINE_AA)
    
    players_location_str = ""
    if players_location == LEFT:
        players_location_str = "(left)"
    elif players_location == RIGHT:
        players_location_str = "(right)"
    cv2.putText(frame, f'ball possession changes: {possession_changes} {players_location_str}',
        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1, cv2.LINE_AA)


# load in grayscale the background image obtained merging the left and the right part of two frames 
# representing the court without people
background_gray = cv2.imread(f'../material/background.png', cv2.IMREAD_GRAYSCALE)

# define two different court mask
mask_points1 = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
court_mask1 = get_mask_from_points(mask_points1, dim=background_gray.shape[:2])

mask_points2 = np.array([(56, 641), (323, 548), (511, 557), (696, 561), (901, 554), (1063, 543), (1353, 630), (1089, 700), (708, 739), (368, 715), (89, 661), (49, 644)], np.int32)
court_mask2 = get_mask_from_points(mask_points2, dim=background_gray.shape[:2])

background_gray = cv2.bitwise_and(background_gray, background_gray, mask=court_mask1)

# define dilatation kernel
dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

# load the yolo v5 detector
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
detector.classes = [0] # find only objects of class 'person'

# initialize the Kalman filter: take into consideration speed and acceleration of the central point of track_window
kalman = cv2.KalmanFilter(6,2)
kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0,0.5,0],[0,1,0,1,0,0.5],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32) * 0.0003
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1


# open the video
cap = cv2.VideoCapture('../material/CV_basket.mp4')
ret, frame = cap.read()
if ret == False:
    sys.exit(1)

# initialize the output video
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, RESOLUTION)

# initialize the track window considering the user selection
print("select the person to track and press ENTER")
track_window = cv2.selectROI('frame', frame, showCrosshair=False)
cx, cy, cw, ch = track_window
tracking_points = [(cx+cw//2, cy+ch//2, False)]

# initialize the parameters used to detect when the ball possession change
avg_right, avg_left = 5, 5 # initially consider 5 players on the left and 5 on the right half-court
possession_changes = 0

frame_index = 0
last_detection_frame = -1
while cap.isOpened() and ret:
    # calculate the motion mask for this frame
    motion_mask = get_motion_mask(frame, background_gray, court_mask1, dilatation_kernel=element)

    # apply yolo to get the predictions
    with torch.no_grad():
        frame = frame[:,:,::-1] # invert color channels BGR to RGB
        results = detector(frame)
        frame = frame[:,:,::-1]

    # split the predictions in valid and bad predictions
    valid_predictions, bad_predictions = filter_predictions(results.xyxy, court_mask2)

    # check if tracking should continue (target not lost)
    if track_window is not None:
        # get the new track_window
        track_window, predicted = track(frame, motion_mask, track_window, valid_predictions+bad_predictions, kalman)
        
        cx, cy, cw, ch = track_window
        tracking_points.append((cx+cw//2, cy+ch//2, predicted))        

        if not predicted:
            last_detection_frame = frame_index
        
        # check if target lost
        if predicted and last_detection is not None and \
                (l2norm(last_detection[:2], track_window[:2]) > MAX_DISTANCE_NO_DETECTION or 
                frame_index-last_detection_frame > MAX_FRAMES_NO_DETECTION):
            print('target lost!')
            track_window = None

    # draw the trajectory of the tracking
    frame = plot_trajectory(frame, tracking_points)

    # count valid predictions in the left and right half-court
    left_counter = 0
    right_counter = 0
    for xmin, ymin, xmax, ymax, confidence, cl in valid_predictions:
        # draw the player bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 2)
        if (xmax+xmin)/2 < frame.shape[1]/2:
            left_counter += 1
        else:
            right_counter += 1
            
    # update estimation of number of people in the left and right half-court
    alpha = 0.8
    avg_left = alpha*avg_left + (1-alpha)*left_counter
    avg_right = alpha*avg_right + (1-alpha)*right_counter

    for xmin, ymin, xmax, ymax, confidence, cl in bad_predictions:
        # draw the player bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BLUE, 2)

    # draw the track_window
    if track_window is not None:
        (xw, yw, ww, hw) = track_window
        if not predicted:
            cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), BROWN, 3)
        else:
            cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), PURPLE, 3)
    
    
    # check if ball possession change (RIGHT to LEFT)
    if players_location != LEFT and avg_right < 0.1:
        players_location = LEFT
        possession_changes += 1
        print('change possession (left)')

    # check if ball possession change (LEFT to RIGHT)
    if players_location != RIGHT and avg_left < 0.1:
        players_location = RIGHT
        possession_changes += 1
        print('change possession (right)')

    # display the number people detected and ball possession information
    put_text(frame, valid_predictions, bad_predictions, players_location, possession_changes)

    # show the processed frame and save it to the output video
    cv2.imshow('frame', frame)
    out.write(frame)


    k = cv2.waitKey(DELAY)
    if k == ord('q'): # quit
        break
    elif k == ord(' '): # pause
        while cv2.waitKey(0) != ord(' '):
            continue

    ret, frame = cap.read()
    frame_index += 1

# release the resources 
cap.release()
out.release()
cv2.destroyAllWindows()