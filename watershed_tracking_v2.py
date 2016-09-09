#load  packages
import cv2
import os
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from PIL import Image
import glob
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import seaborn as sns
import random
import sys
import time
from skimage.transform import rotate
from skimage.color.colorlabel import label2rgb
import pylab as py
import csv
from skimage.measure import structural_similarity as ssim
from IPython import display
from sklearn.preprocessing import normalize

COLORS = [(139, 0, 0),
          (0, 100, 0),
          (0, 0, 139)]

def random_color():
    return random.choice(COLORS)


def filereader(filesdirectory):

    cap = cv2.VideoCapture(filesdirectory)
    timestamp = []
    frames = []
    try:
        while cap.isOpened():
            _, img = cap.read()
            # get the frame in seconds
            t1 = cap.get(0)

            if img is None:
                break
            frames.append(img)
            timestamp.append(t1)
    except EOFError:
        pass
    return timestamp, frames


def watershedtracking(frame):

    xs, ys, CID = [], [], []

    try:
        image = frame
        # pre-process the image before performing watershed
        # load the image and perform pyramid mean shift filtering to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(image, 10, 39)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,
             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=10,
         labels=thresh)

        # # perform a connected component analysis on the local peaks,
        # # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)


        print("[INFO] {} unique contour found".format(len(np.unique(labels)) - 1))

        #  loop over the unique labels returned by the Watershed
        # algorithm for finding the centriod cx and cy
        # of each contour Cx=M10/M00 and Cy=M01/M00.

        for (i, c) in enumerate(labels):
            # draw the contour
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == i] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # dilation = cv2.dilate(mask, kernel, iterations=2)
            # Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            cv2.imshow('showme', mask)
            cv2.waitKey(0)

            # detect contours in the mask and grab the largest one
            #cv2.imshow('mask', mask)
            #cv2.waitKey(0)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            #c = max(cnts, key=cv2.contourArea)
            for c in cnts:
                # get the center of the object
                ((x, y), _) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)

                xs.append(x)
                ys.append(y)
                CID.append(i)
            if i >0:
                break

    except EOFError:
        pass

    return xs, ys


def datamanipulation():
    x_axis, y_axis, frameID, tseries, data = [], [] , [], [],[]
    #path = '/home/sami/9K5F98PS_F00000011.avi'
    path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
    path = '/home/sami/9K5F98PS_F00000011.avi'
    t, f = filereader(path)

    for (i,frame) in enumerate(f):
        xs, ys = watershedtracking(frame)
#        a, b = xs.ravel()  # tmp new value
       # c, d = ys.ravel()  # tmp old value

        x_axis.append(xs)
        y_axis.append(ys)
        frameID.append(i)
        tseries.append(t[i])

        if i > 0:
            print xs[0], xs[-1]
            print ys[0], ys[-1]
            cv2.line(frame, (int(xs[0]), int(ys[0])), (int(xs[-1]), int(ys[-1])), (255, 0, 255), 1)
            cv2.imshow('display', frame)
            cv2.waitKey(0)

#        a, b = new.ravel()  # tmp new value
       # c, d = old.ravel()  # tmp old value


datamanipulation()

''' kernel = np.ones((3, 3), np.uint8)
 opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
 # sure background area
 sure_bg = cv2.dilate(opening, kernel, iterations=3)

 # Finding sure foreground area
 dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
 ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

 # sure background area
 sure_bg = cv2.dilate(opening, kernel, iterations=3)
 sure_fg = np.uint8(sure_fg)
 unknown = cv2.subtract(sure_bg, sure_fg)

 cv2.imshow('mask', sure_fg)
 cv2.waitKey(0)
 exit()'''