
#load  packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import seaborn as sns
import glob
import random
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

COLORS = [(139, 0, 0),
          (0, 100, 0),
          (0, 0, 139)]

rgbint = 129
def random_color():
    return random.choice(COLORS)

def VideoFrameReaders(VideoDirectory):
    cap = cv2.VideoCapture(VideoDirectory)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    timestamp = []
    count = 0
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            time = cap.get(0) #get the frame in seconds
            timestamp.append(time)

            print timestamp

            if frame == None:
                break;
           # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            image = frame.reshape((frame.shape[0]*frame.shape[1],3))
            K = 4
            clf = MiniBatchKMeans(K)

            #predict cluster labels and quanitize each color based on the labels

            cls_labels = clf.fit_predict(image)
            print cls_labels
            cls_quant = clf.cluster_centers_astype("uint8")[labels]


    except EOFError:
        pass
path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');
VideoFrameReaders(path[0])