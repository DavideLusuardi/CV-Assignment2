import torch
import cv2
import numpy as np
import bg_subtraction

mask_points = np.array([(56, 641), (323, 548), (511, 557), (696, 561), (901, 554), (1063, 543), (1353, 630), (1089, 700), (708, 739), (368, 715), (89, 661), (49, 644)], np.int32)
mask = bg_subtraction.get_mask(mask_points)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0] # filter for person

RIGHT = 0
LEFT = 1
players_location = None

def filter_predictions(predictions, mask):
    good_predictions = list()
    bad_predictions = list()
    for xmin, ymin, xmax, ymax, confidence, cl in predictions:
        if mask[(ymin+ymax)//2][(xmin+xmax)//2] == 255:
            good_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))
        else:
            bad_predictions.append((xmin, ymin, xmax, ymax, confidence, cl))

    return good_predictions, bad_predictions

i = 0
cap = cv2.VideoCapture('../material/CV_basket.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break

    # frame = frame[:,:,::-1]
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.resize(frame, (1280, 720))
    # frame = cv2.resize(frame, (1080, 810))

    i += 1
    if i % 10 != 0:
        continue
    with torch.no_grad():
        results = model(frame)
    # for i, img in enumerate(results.render()):
    #     cv2.imshow('res', img)
    #     # print(results.xyxy[i])

    left_counter = 0
    right_counter = 0
    predictions = list()
    for result in results.xyxy:
        for xmin, ymin, xmax, ymax, confidence, cl in result:
            predictions.append((int(xmin.cpu()), int(ymin.cpu()), int(xmax.cpu()), int(ymax.cpu()), float(confidence.cpu()), float(cl.cpu())))
    
    good_predictions, bad_predictions = filter_predictions(predictions, mask)
    # print('good preds', good_predictions)
    # print('bad preds', bad_predictions)
    for xmin, ymin, xmax, ymax, confidence, cl in good_predictions:
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        if (xmax+xmin)/2 < frame.shape[1]//2:
            left_counter += 1
        else:
            right_counter += 1
    for xmin, ymin, xmax, ymax, confidence, cl in bad_predictions:
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    

    # frame = frame[:,:,::-1]
    cv2.imshow('frame', frame)

    print('-----------------------------')
    print(players_location)
    print(left_counter, right_counter)
    if right_counter == 0 and players_location != LEFT:
        players_location = LEFT
        print('change possession to left')
        cv2.waitKey(0)

    if left_counter == 0 and players_location != RIGHT:
        players_location = RIGHT
        print('change possession to right')
        cv2.waitKey(0)    

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue

cap.release()
cv2.destroyAllWindows()