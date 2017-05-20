from moviepy.editor import VideoFileClip, clips_array
import car_detection
from sklearn.externals import joblib
import numpy as np
import cv2
def generate_heatmap(img):
    windows = []
    svc = joblib.load('svc.pkl')
    X_scaler = joblib.load('X_scaler.pkl')

    ystart = 350
    ystop = 656
    scale = 1.5
    fe = car_detection.FeatureExtractor()

    for current_scale in [1.0, 1.33, 1.66, 2.0]:
        windows.extend(fe.find_cars(img, ystart, ystop, current_scale, svc, X_scaler))

    heatmap = np.zeros_like(img)

    for window in windows:
        #cv2.rectangle(draw_img,window[0], window[1],(0,0,255),6)
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
