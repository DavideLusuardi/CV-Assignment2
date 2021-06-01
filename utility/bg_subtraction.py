import cv2
import numpy as np


def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    # w, h = 1220, 287
    return x_min, y_min, x_max-x_min, y_max-y_min

def get_mask(mask_points, dim=None):
    if dim is None:
        dim = background.shape[:2]
    x,y,w,h = get_box(mask_points)
    mask_points = mask_points.reshape((-1,1,2))
    mask = np.zeros(dim, np.uint8)
    cv2.fillPoly(mask,[mask_points],255)
    return mask

root_dir = '..'
background = cv2.imread(f'{root_dir}/material/background.png', cv2.IMREAD_GRAYSCALE)

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
mask = get_mask(mask_points)
# mask = mask[y:y+h, x:x+w]

# background = background[y:y+h, x:x+w]
background = cv2.bitwise_and(background, background, mask=mask)

dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

def subtract(frame):
    # frame_cut = frame[y:y+h, x:x+w]
    #color conversion
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask) # apply mask

    #bg subtraction
    diff = cv2.absdiff(background, frame_gray)
    #mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, element, iterations=1)

    frame_masked = cv2.bitwise_and(frame, frame, mask=motion_mask)

    return frame_masked, motion_mask