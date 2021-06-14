# Computer Vision - Assignment 2
This project code extracts three basic analytics from the video [CV_basket.mp4](https://drive.google.com/file/d/1AzextISOXsr6VysY3CkYtyyHBmXJA7js/view) and produce an output video `output.avi` showing the results.

The following three main tasks are performed:
1. run the yolo v5 detector on the whole video reporting at each frame the number of people detected in total and in the court
1. track the user selected person and plot the trajectory until the person is lost
1. detect how many times the ball possession changes

## Requirements
Python >= 3.6.0 is required with the following dependencies installed:
* OpenCV 4.5.2
* Pytorch
* Torchvision
* NumPy

[Yolo v5](https://github.com/ultralytics/yolov5) has been used as detector. For this reason its [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies must be installed.

The input video [CV_basket.mp4](https://drive.google.com/file/d/1AzextISOXsr6VysY3CkYtyyHBmXJA7js/view) should be placed in the project directory.

## Usage
The code can be run typing the command from the project directory:
```bash
python assignment2.py
```

## Workflow
The following operations are performed for the three tasks.

### People detection
For each video frame the people are detected by the neural network [Yolo v5](https://github.com/ultralytics/yolov5) that is implemented in pytorch. More information can be found on the official [website](https://ultralytics.com/yolov5).
The function `filter_predictions()` splits the predictions in two disjoint sets: valid and bad predictions.
In order to decide if a prediction is valid, a mask representing the court is used: a prediction is considered valid if the center of the bounding box is positioned in the court (i.e. the value of the center point in the mask is 255).

The valid predictions are shown in red in the output video whereas the bad predictions are in blue.
The number of total and valid predictions are displayed in the top-left corner. 

### Person tracking
Initially, the user should select, with a rectangular selection, a person in the scene to track.

The `background.png` image, representing the court without people, is located in the project directory and has been previously obtained merging the left and the right half part of two different frames where the court was empty from people.

A motion mask of each frame is calculated by the function `get_motion_mask()` that apply the background subtraction between the frame and the background image.
This mask is used afterwards to calculate the normalized histogram of the tracked person.

The function `track()` is called on each frame. This function performs the tracking of the person and calculate the new `track_window`.
Starting from the predictions previously founded by Yolo, the function calculates two metrics in order to find the best prediction:
* the Euclidean distance between the center of each predcition and the center of the previous `track_window`;
* the Bhattacharyya distance between the normalized histogram of the person and the prediction.
The prediction that minimizes the weighted sum of the two distances is considered the best prediction of the tracking window.
If this weighted distance is greater than a dynamic threshold, then the prediction is discarded and the `track_window` is estimated using a Kalman filter.
The dynamic threshold is calculated taking into consideration the last frame in which the `track_window` was equal to a prediction of Yolo (i.e. not estimated by the Kalman filter).

The Kalman filter is designed to estimate the speed and the acceleration of the central point of the `track_window`.

At each frame, the `track_window` is drawn and the trajectory is plotted on the frame: in purple when it is estimated by the Kalman filter, in brown otherwise. 

### Ball possession
In order to count how many times the ball possession changes (i.e. all 10 players go from one half of the court to the other one), an estimation of the number of people located in the two halves of the court is calculated.

The left half-court estimation has been calculated by the following formula:
```
avg_left = alpha*avg_left + (1-alpha)*left_counter
```
where `alpha = 0.8` and `left_counter` is the number of valid predictions in the left half-court (initially `5`).
The right half-court estimation has the same form.

The ball possession changes when the players are all in one half of the court and previously they were in the opposite half.