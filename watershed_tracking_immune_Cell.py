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
import colorsys
import time
from skimage.transform import rotate
from skimage.color.colorlabel import label2rgb
import pylab as py
import csv
from skimage.measure import structural_similarity as ssim
from sklearn.preprocessing import normalize



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


def watershedtracking(frame,d):
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

        #  loop over the unique labels returned by the Watershed
        # algorithm for finding the centriod cx and cy
        # of each contour Cx=M10/M00 and Cy=M01/M00.
        # generate color according to the number of identified cells
        N = len(np.unique(labels))+1

        HSV_tuples = [(B * 2.0 / N, 0.6, 0.6) for B in range(N)]
        RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)


        for ii, c in enumerate(labels):
            # draw the contour
            if ii > 0:
                try:
                    mask = np.zeros(gray.shape, dtype="uint8")
                    mask[labels == ii] = 255
                    # detect contours in the mask and grab the largest one
                    cv2.imwrite('/home/sami/Desktop/code/segmentation/immune_cell_seg/Immune2_%d.png' % ii, mask)
                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    # assign a color for each cell
                    color2 = RGB_tuples[ii]
                    color3 = tuple([256 * t for t in color2])
                    cv2.drawContours(image, cnts, -1, color3, 1)
                    cv2.imshow("Contour", image)
                    cv2.waitKey(33)
                except IndexError:
                    continue
        cv2.imwrite('/home/sami/Desktop/code/segmentation/immune_cell_seg/drawContour2/Image_%d.png' % d, image)
        cv2.destroyAllWindows()
    except EOFError:
        passs

    return xs, ys


def datamanipulation():
    x_axis, y_axis, frameID, tseries, data = [], [] , [], [],[]
    path = '/home/sami/9K5F98PS_F00000011.avi'
    t, f = filereader(path)


    for (i,frame) in enumerate(f):
        xs, ys = watershedtracking(frame, i)

        x_axis.append(xs)
        y_axis.append(ys)
        frameID.append(i)
        tseries.append(t[i])

    data.append([frameID, x_axis, y_axis, tseries])
    data = data[0]
    unpacked = zip(data[0], data[1], data[2], data[3])

    with open('watershed_CANCER3.0_' + str(i) + '.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'X_points','Y_points', 'TimeSeries',))
        for value in unpacked:
            writer.writerow(value)

datamanipulation()
