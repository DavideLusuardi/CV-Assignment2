'''
    Extract clean background from video.
'''

import cv2
import numpy as np

court_left = cv2.imread('../material/court_left.png', cv2.IMREAD_COLOR)
court_right = cv2.imread('../material/court_right.png', cv2.IMREAD_COLOR)

if court_left is None or court_right is None:
    print("error reading images")

h1, w1 = court_left.shape[:2]
h2, w2 = court_right.shape[:2]

assert h1 == h2 and w1 == w2

center = w1//2
# background = np.zeros((h1, w1, 3), np.uint8)
background = court_left.copy()
background[:h2, center:, :3] = court_right.copy()[:h2, center:, :3]

cv2.imshow('background', background)


mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(background.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)

backgroundMasked = cv2.bitwise_and(background, background, mask=mask)
cv2.imshow('backgroundMasked', backgroundMasked)

k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('../material/background.png', background)

cv2.destroyAllWindows()