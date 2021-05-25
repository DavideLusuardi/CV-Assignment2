import torch
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
# model.to(device)

model.classes = [0] # filter for person

# Images
# dir = 'yolov5/data/images/'
# imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

# # Inference
# results = model(imgs)
# results.print()  # or .show(), .save()
# results.save()

cap = cv2.VideoCapture('CV-Assignment2/material/CV_basket.mp4')
frames = list()
for i in range(1000):
    ret, frame = cap.read()
    frame = frame[:,:,::-1]
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.resize(frame, (1280, 720))
    frame = cv2.resize(frame, (1080, 810))

    with torch.no_grad():
        results = model(frame, size=1080)
    # for i, img in enumerate(results.render()):
    #     cv2.imshow('res', img)
    #     # print(results.xyxy[i])

    for predictions in results.xyxy:
        for xmin, ymin, xmax, ymax, confidence, cl in predictions:
            frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,0), 2)

    frame = frame[:,:,::-1]
    cv2.imshow('frame', frame)

    k = cv2.waitKey(50)
    if k == ord('q'):
        break
    elif k == ord(' '):
        while cv2.waitKey(0) != ord(' '):
            continue

cv2.destroyAllWindows()