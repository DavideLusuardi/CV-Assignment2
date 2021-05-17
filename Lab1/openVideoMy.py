import cv2

# open the video path
# cap = cv2.VideoCapture("../material/Video.mp4")

cap = cv2.VideoCapture(0)

for i in range(10):
    # capture frame by frame
    ret, frame = cap.read()

    cv2.imwrite('img'+str(i)+".jpg", frame)

    # display video
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

# release resources of capture
cap.release()