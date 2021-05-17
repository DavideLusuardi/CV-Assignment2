import cv2

image = cv2.imread("../material/Google.jpg", 1)

cv2.imshow("hello world", image)
cv2.waitKey(0)

