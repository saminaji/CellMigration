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

count = 0
def watershedtracking(frame, count):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    xs, ys, CID = [], [], []

    try:
        image = frame
        cv2.imwrite('/home/sami/Desktop/code/Foo/OrgOTSU_%d.png' % int(count), image)
        count += 1
        fgmask = fgbg.apply(image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        # pre-process the image before performing watershed
        # load the image and perform pyramid mean shift filtering to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(image, 10, 39)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,
             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('OTSU', thresh)
        cv2.imwrite('/home/sami/Desktop/code/segmentation/cancer_cell_seg//OTSU_%d.png' % int(1), thresh)
        cv2.waitKey(33)
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

            # detect contours in the mask and grab the largest one
            #cv2.imshow('mask', mask)
           # cv2.imwrite('/home/sami/Desktop/code/segmentation/immune_cell_seg/Cancer3_%d.png'% i, image)
           # cv2.waitKey(0)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            N = len(np.unique(labels))

            count = 0
            color33 = 200* [(255,0,0),(255,255,0), (0,255,255),(0,0,255), (255,105,180)]

            color3 = color33[i]
            print color3
            # ((x, y), _) = cv2.minEnclosingCircle(c)

            # cv2.circle(image,(int(x),int(y)),2,color3,6)#frame

            # cv2.drawContours(image, c, -1, color3, 5)
            c2 = [ c for  c in cnts]
            for c2 in cnts:
                ((x, y), _) = cv2.minEnclosingCircle(c2)
                cv2.drawContours(image, c2, -1, color3, 3)
                cv2.imshow("Contour", image)
                cv2.putText(image,'(%d, %d)', (int(x) % int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.2,color3,2)
                cv2.imwrite('/home/sami/Desktop/code/Foo/F22_%d.png' % i, image)
                cv2.waitKey(33)


            HSV_tuples = [(B * 120.0 / N, 127.0, 123.0) for B in range(N)]
            RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)
            #c = max(cnts, key=cv2.contourArea)
            '''count
            for cc, c in enumerate(cnts):
               # print cc
                # get the center of the object
                #color2 = RGB_tuples[cc]
               # color3 = tuple([256 * t for t in color2])
                color3 = color33[cc]
                print color3
               # ((x, y), _) = cv2.minEnclosingCircle(c)

                #cv2.circle(image,(int(x),int(y)),2,color3,6)#frame

               # cv2.drawContours(image, c, -1, color3, 5)
                cv2.drawContours(image, c, -1, color3, 3 )
                cv2.imshow("Contour", image)
                cv2.imwrite('/home/sami/Desktop/code/segmentation/immune_cell_seg/drawContour/Image3_%d.png' % cc, image)
                cv2.waitKey(0)'''
                #xs.append(x)
                #ys.append(y)
               # CID.append(i)
           # vis = np.concatenate((mask, image), axis=0)
            #cv2.imwrite('home/sami/Desktop/code/segmentation/cancer_cell_seg/drawContour/Sidebyside_%d.png' % i, image)
            cv2.waitKey(33)
    except EOFError:
        passs

    return xs, ys


def datamanipulation():
    x_axis, y_axis, frameID, tseries, data = [], [] , [], [],[]
    path = '/home/sami/9K5F98PS_F00000011.avi'
    #path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
    t, f = filereader(path)

    for (i,frame) in enumerate(f):
        if i == 0:
            continue
        xs, ys = watershedtracking(frame, count)

        x_axis.append(xs)
        y_axis.append(ys)
        frameID.append(i)
        tseries.append(t[i])

    data.append([frameID, x_axis, y_axis, tseries])
    data = data[0]

#    unpacked = zip([frameID, x_axis,y_axis,tseries])

    unpacked = zip(data[0], data[1], data[2], data[3])

    with open('watershed_CANCER2.0_' + str(i) + '.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'X_points','Y_points', 'TimeSeries',))
        for value in unpacked:
            writer.writerow(value)
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