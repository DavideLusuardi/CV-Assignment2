import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_box(mask_points):
    x_min = min(mask_points, key=lambda e: e[0])[0]
    y_min = min(mask_points, key=lambda e: e[1])[1]
 
    x_max = max(mask_points, key=lambda e: e[0])[0]
    y_max = max(mask_points, key=lambda e: e[1])[1]

    # w, h = 1220, 287
    return x_min, y_min, x_max-x_min, y_max-y_min

root_dir = '..'
background = cv2.imread(f'{root_dir}/material/background.png', cv2.IMREAD_GRAYSCALE)

mask_points = np.array([(87, 660), (298, 708), (494, 732), (709, 740), (902, 728), (1114, 696), (1307, 646), (1243, 540), (1084, 458), (996, 531), (696, 539), (389, 534), (313, 453), (131, 542)], np.int32)
x,y,w,h = get_box(mask_points)
mask_points = mask_points.reshape((-1,1,2))
mask = np.zeros(background.shape[:2], np.uint8)
cv2.fillPoly(mask,[mask_points],255)
mask = mask[y:y+h, x:x+w]

background = background[y:y+h, x:x+w]
background = cv2.bitwise_and(background, background, mask=mask)


def compare_histograms(images, masks=None):
    hists = list()
    for i, img in enumerate(images):
        mask = masks[i] if masks is not None else None
        cv2.imshow(f'selection {i}', img)
        if mask is not None:
            cv2.imshow(f'mask {i}', mask)

        hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8],
                [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)

    # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        print(f'{methodName}')
        # initialize the results dictionary and the sort
        # direction
        results = [[None]*len(hists)]*len(hists)
        reverse = False
        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        # loop over the index
        for i, hist1 in enumerate(hists):
            for j, hist2 in enumerate(hists):
                # compute the distance between the two histograms
                # using the method and update the results dictionary
                d = cv2.compareHist(hist1, hist2, method)
                # print(f"distance {d}")
                results[i][j] = d
            print(results[i])


dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

masks = list()
images = list()
cap = cv2.VideoCapture(f"{root_dir}/material/CV_basket.mp4")
for i in range(1000):
    ret, frame = cap.read()
    frame_cut = frame[y:y+h, x:x+w]
    #color conversion
    frame_gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask) # apply mask

    #bg subtraction
    diff = cv2.absdiff(background, frame_gray)
    #mask thresholding
    ret2, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    # motion_mask = cv2.dilate(motion_mask, element, iterations=1)

    frame_masked = cv2.bitwise_and(frame_cut, frame_cut, mask=motion_mask)

    if i == 0:
        roi = cv2.selectROI('full_img', frame_cut, showCrosshair=False)
        xr,yr,wr,hr = roi
        img = frame_cut[yr:yr+hr, xr:xr+wr]
        m = motion_mask[yr:yr+hr, xr:xr+wr]
        images.append(img)
        masks.append(m)
    else:
        for j in range(2):
            roi = cv2.selectROI('full_img', frame_cut, showCrosshair=False)
            xr,yr,wr,hr = roi
            img = frame_cut[yr:yr+hr, xr:xr+wr]
            m = motion_mask[yr:yr+hr, xr:xr+wr]
            images.append(img)
            masks.append(m)

        break

compare_histograms(images, masks)
cv2.waitKey(0)