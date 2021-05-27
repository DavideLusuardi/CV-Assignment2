import torch
import cv2
import numpy as np

def dist2(c1, c2):
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2


WIDTH = 128
HEIGHT = 256
def trackWindow(window, width=128, height=256, origin=(0,0)):
    # TODO: check not going out of borders
    (xw, yw, ww, hw) = window
    xc, yc = xw+ww//2, yw+hw//2
    x = xc-width//2 + origin[0]
    y = yc-height//2 + origin[1]
    # return frame[y:y+height, x:x+width], (x,y,width,height), (width//2, height//2)
    return (x,y,width,height)


def track(frame, track_window, model):
    (xw, yw, ww, hw) = track_window
    frame_window = frame[yw:yw+hw, xw:xw+ww]
    track_center = (ww//2, hw//2)
    # track_center = (frame_window.shape[1]//2, frame_window.shape[0]//2)

    with torch.no_grad():
        results = model(frame_window)
        for img in results.render():
            cv2.imshow('render',img)
    

    for predictions in results.xyxy:
        for xmin, ymin, xmax, ymax, confidence, cl in predictions:
            center = ((xmax+xmin)/2, (ymax+ymin)/2)
            x1p, x2p, y1p, y2p = int(xmin), int(xmax), int(ymin), int(ymax)

            distance = dist2(track_center, center)
            # TODO: calculate histogram distance


    print(f"distance: {min_dist}")
    if min_dist < 1000:
        track_window = trackWindow((x1p, y1p, x2p-x1p, y2p-y1p), origin=(xw, yw))
        print("correction",np.array([[np.float32(track_window[0])],[np.float32(track_window[1])]]))
        kalman.correct(np.array([[np.float32(track_window[0])],[np.float32(track_window[1])]]))
        print("useless prediction", kalman.predict())
        predicted = False
    else:        
        current_pre = kalman.predict()
        # print("prediction",current_pre)
        cpx, cpy = current_pre[0].astype(int).item(), current_pre[1].astype(int).item()
        track_window = (cpx, cpy, WIDTH, HEIGHT)
        predicted = True
    
    return track_window, predicted



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model.classes = [0] # filter for person


kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1

# kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]],np.float32)
# kalman.measurementMatrix = 1. * np.ones((1, 2),np.float32)
# kalman.processNoiseCov = 1e-5 * np.eye(2)
# kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1),np.float32)
# kalman.errorCovPost = 1. * np.ones((2, 2))
# kalman.statePost = 0.1 * np.random.randn(2, 1)


cap = cv2.VideoCapture('../material/CV_basket.mp4')
for i in range(1000):
    ret, frame = cap.read()
    # frame = frame[:,:,::-1]
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.resize(frame, (1280, 720))
    # frame = cv2.resize(frame, (1080, 810))

    if i == 0:
        roi = cv2.selectROI('frame', frame, showCrosshair=False)
        # x1, x2, y1, y2 = x, x+w, y, y+h
        # track_center = ((x2+x1)//2, (y2+y1)//2)
        track_window = trackWindow(roi)
        # track_center = (xw+ww//2, yw+hw//2)

    # elif i%5 == 0:
    track_window, predicted = track(frame, track_window, model)
    print(track_window)
    print(f"predicted {predicted}")

    (xw, yw, ww, hw) = track_window
    if not predicted:
        cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), (255,0,0), 3)
    else:
        cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), (255,255,255), 3)
    

    # frame = frame[:,:,::-1]
    cv2.imshow('frame', frame)

    k = cv2.waitKey(40)
    if k == ord('q'):
        break
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue

cv2.destroyAllWindows()