# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

from numpy.lib.type_check import imag


def detect(args, image, net, LABELS):
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	size = (416, 416) # original one
	size = (512, 512)
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size,
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	objects = [(boxes[i], classIDs[i], LABELS[classIDs[i]], confidences[i]) for i in idxs.flatten()]
	return objects

def draw_labels(image, objects, COLORS):
	for box, classID, label, confidence in objects:
		(x, y) = (box[0], box[1])
		(w, h) = (box[2], box[3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classID]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(label, confidence)
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

	return image


def load_data(args):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return net, LABELS, COLORS

def process_video(args, net, LABELS, COLORS):
    cap = cv2.VideoCapture(args['video'])
    for i in range(1000):
        # Capture frame-by-frame
        ret, frame = cap.read()

        objects = detect(args, frame, net, LABELS)
        frame = draw_labels(frame, objects, COLORS)

        # show the output image
        cv2.imshow("Image", frame)

        k = cv2.waitKey(1)
        if k == ord('q'): #close video if q is pressed
            break

def process_image(args, image, net, LABELS, COLORS):
	objects = detect(args, image, net, LABELS)
	image = draw_labels(image, objects, COLORS)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

def main(args):
	net, LABELS, COLORS = load_data(args)

	if args['image'] != None:
		image = cv2.imread(args["image"])
		process_image(image)
	elif args['video'] != None:
		process_video(args, net, LABELS, COLORS)


if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", default=None,
		help="path to input video")
	ap.add_argument("-i", "--image", default=None,
		help="path to input image")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.3,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.2,
		help="threshold when applying non-maxima suppression")
	args = vars(ap.parse_args())

	main(args)
