#load  packages
import cv2
import os
import matplotlib.pyplot as plt
import Image
import glob
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import random
import sys
from scipy.spatial import cKDTree
from skimage.color.colorlabel import label2rgb
import pylab as py
import csv
from IPython import display
import Image


def kalman_xy(x, P, measurement, R,
              motion=np.matrix('0. 0. 0. 0.').T,
              Q=np.matrix(np.eye(4))):
    """
    Parameters:
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    print measurement
    return kalman(x, P, measurement, R, motion, Q,
                  F=np.matrix('''1. 0. 1. 0.;
                          0. 1. 0. 1.;
                          0. 0. 1. 0.;
                          0. 0. 0. 1.
                          '''),       H=np.matrix('''
                          1. 0. 0. 0.;
                          0. 1. 0. 0.'''))


def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I  # Kalman gain
    x = x + K * y
    I = np.matrix(np.eye(F.shape[0]))  # identity matrix
    P = (I - K * H) * P

    # PREDICT x, P based on motion
    x = F * x + motion
    P = F * P * F.T + Q

    return x, P


def filereader(filesdirectory):
    cap = cv2.VideoCapture(filesdirectory)
    timestamp = []
    frames = []
    try:
        while cap.isOpened():
            ret, img = cap.read()
            # get the frame in seconds
            t1 = cap.get(0)
            timestamp.append(t1)
            if img is None:
                break
            frames.append(img)
    except EOFError:
        pass
    return timestamp, frames

def readtiff(path):
    frames = []
    tif = TIFF.open(path, mode='r')

    try:
        for cc, tframe in enumerate(tif.iter_images()):
            frames.append(tframe)
    except EOFError:
        pass
    return frames
tt = 0
resultX, resultY = [], []

def watershedsegmentation(frame, resultX, resultY, tt):
    contors = []
    try:

       # im = cv2.threshold(frame, 173, 255, cv2.THRESH_BINARY)
       # im = im[1]
       # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       # dilation = cv2.dilate(im, kernel, iterations=2)
        #Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        #closing = cv2.morphologyEx(Gradient, cv2.MORPH_CLOSE, kernel)
        # pre-process the image before performing watershed
        # load the image and perform pyramid mean shift filtering to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(frame, 10, 39)
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
        result = []

        masks2 = np.zeros(gray.shape, dtype="uint8")
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it

            if label == 0:
                 continue

            #otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            #cv2.imshow('mask', mask)
           # cv2.waitKey(0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


            #cv2.imshow('closed', mask)
           # cv2.waitKey(0)
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)[-2]

            try:
                cnt = cnts[0]
                contors.append(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 1)
                cv2.putText(frame,('ID %d'% tt),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, 255 )
                result.append((x,y))

                cv2.waitKey(33)
            except IndexError:
                continue

        tt += 1
        if tt < 180:
            cv2.imwrite('/home/sami/Desktop/code/segmentation/2016_7_4_12_27/%d.png' % tt, masks2)




        #cv2.drawContours(frame, cnt, -1, (255,0,255), 2)
        # cv2.imshow('f', frame)
        cv2.imwrite('Image_%d.png' % 1 , frame)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            # break

    except EOFError:
        pass
    return contors, mask, result

def avarage(x1, x2):

    return x1+x2 / 2

# main function



if __name__=='__main__':

    # read frames of the video
    path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')
    cellCentroid = []
    path = '/home/sami/9K5F98PS_F00000011.avi'

    # path = glob.glob('/home/sami/Desktop/movies/movies/*.avi')

    for p in range(len(path[0])):
        t, f = filereader(path)

        NextFrame, _, resultX = watershedsegmentation(f[0],resultX, resultY,1)
        for iii, xx in enumerate(resultX):

            for ii, frame2 in enumerate(f[1:]):

                NextFrame, _, result2 = watershedsegmentation(frame2, resultX, resultY, ii)
                tree = cKDTree(result2)

                dists, indexes = tree.query(np.array(xx), k=3)
                for dist, index in zip(dists, indexes):
                    if dist <= 5:
                        resultY.append(())
                        print 'distance %f:  %s' % (dist, result2[index])

                    if dist > 5:
                        if len(resultY) > 2:
                            centX = resultY[index]

                        if len(resultY) < 2:
                            x,y = resultY[index]
                            centX = avarage(x,y)
                    exit()



        else:
            dist = numpy.linalg.norm()
            cv2.line(masks2, (int(x), int(y)), (int(resultX[0]), int(resultY[0])), (255, 0, 255), 2)
            cv2.imshow('line', masks2)
            resultX.append(int(x))
            resultY.append(int(y))

    '''unpacked = zip(cc[0], cc[1], cc[2], cc[3], cc[4])
    with open('data_Centroid_tiff2' + str(count) + '.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'CellID', 'x-axis', "y-axis", 'time',))
        for value in unpacked:''
            writer.writerow(value)'''


