

#load  packages
import cv2
import os
import matplotlib as plt
import Image
import glob
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import seaborn as sns
from time import sleep
import random
from skimage.color.colorlabel import label2rgb

COLORS = [(139, 0, 0),
          (0, 100, 0),
          (0, 0, 139)]

rgbint = 129
def random_color():
    return random.choice(COLORS)

def VideoFrameReaders(VideoDirectory):
    cap = cv2.VideoCapture(VideoDirectory)
   # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    timestamp = []
    count = 0
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            time = cap.get(0) #get the frame in seconds
            timestamp.append(time)



            if frame == None:
                break;
            image = frame
            fgmask = fgbg.apply(image)
            #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            #fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
            fgmask = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15);
            # pre-process the image before performing watershed
            # load the image and perform pyramid mean shift filtering to aid the thresholding step
            edged = cv2.Canny(fgmask,123,255)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


            cv2.imshow("input", frame)
            cv2.imshow("Canny", edged)
            cv2.imshow("Closed", closed)
            # find contours (i.e. the 'outlines') in the image and initialize the
            # total number of books found
            #cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total = 0


            sleep(5)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    except EOFError:
        pass
    return count,timestamp,

path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');

c,t = VideoFrameReaders(path[0])



# get frames from a directory
#frames = glob.glob('/home/sami/Desktop/movies/extractFrames/*.jpg')
