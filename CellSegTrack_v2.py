#load  packages
import cv2
import os
from libtiff import TIFF
import matplotlib.pyplot as plt
import Image
import glob
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import random
import sys
import time
import colorsys
from skimage.color.colorlabel import label2rgb
import pylab as py
import csv
from IPython import display
import datatime

# The algorithm aims at performing segmentations and tracking by means of watershed and plus morphological shape operations.
# Here we apply several morphological such as deliation, closing, and gradient 

# create a directory for each submission
now = datetime.datetime.now()
dirLast = str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)
os.mkdir(dirLast)


def filereader(filesdirectory):
    cap = cv2.VideoCapture(filesdirectory)
    timestamp = []
    frames = []
    try:
        while cap.isOpened():
            ret, img = cap.read()
            # get time when the frame captured
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

def watershedsegmentation(frame, N1, nf):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    contors = []

    try:
        # pre-process the image before performing watershed
        # load the image and perform  deliation, gradient, closing, and pyramid mean shift filtering to aid the thresholding step
           # apply a threshold
        im = cv2.threshold(frame, 173, 255, cv2.THRESH_BINARY)
        im = im[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation = cv2.dilate(im, kernel, iterations=2)
        Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        closing = cv2.morphologyEx(Gradient, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('/dirLast/Gradient_%d.png' % N1, closing)

        shifted = cv2.pyrMeanShiftFiltering(closing, 10, 20)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20,
                                  labels=thresh)

        #  perform a connected component analysis on the local peaks,
        #  using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        #  loop over the unique labels returned by the Watershed
        # algorithm for finding the centriod cx and cy
        # of each contour Cx=M10/M00 and Cy=M01/M00. Generate different colors for each cell
        N = len(np.unique(labels))
        HSV_tuples = [(B * 2.0 / N, 0.6, 0.6) for B in range(N)]
        RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)
        mask = np.zeros(gray.shape, dtype="uint8")
        count2 = 0
        NumFiles = len(glob.glob('/dirLast/*.png'))

        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            mask = np.zeros(gray.shape, dtype="uint8")
            if label == 0:
                continue

            # otherwise, allocate memory for the label region and draw
            # it on the mask
            color2 = RGB_tuples[count2]
            color3 = tuple([256 * t for t in color2])
            mask[labels == label] = 255
            #   detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            # get the largest contour, ignore small contour
            areas = [cv2.contourArea(c) for c in cnts]
            max_index = np.argmax(areas)
            cnt = cnts[max_index]
            contors.append(cnt)
            # save some samples for further analysis
            if NumFiles < N1:
                cv2.drawContours(frame, cnt, -1, color3, 2)
                cv2.imwrite('/dirLast/Image_%d.png' % nf, frame)

    except EOFError:
        pass
    return contors, mask

# main function

if __name__=='__main__':

    # read frames of the video
    #path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')
    # path = '/home/sami/9K5F98PS_F00000011.avi'
   # path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')

    CellCentroid = []
    # path = '/home/sami/9K5F98PS_F00000011.avi'
    # path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
    path = '/home/sami/Downloads/ctr_multipage.tif'
    # path = glob.glob('/home/sami/Desktop/movies/movies/*.avi')

    for p in range(len(path[0])):

        if path[-3] == 'tif':
            f = readtiff(path)
        else:
            _, f = filereader(path)
        frameNum = -1
        fNum = len(f)
        elipse = 8
        refFrame, _ = watershedsegmentation(f[0], fNum, frameNum)

        frameID, period, CloseCells, XTrajectory, YTrajectory, CID = [], [], [], [], [], []
        for index2, refCont in enumerate(refFrame):
            centroid = []
            M = cv2.moments(refCont)
            divisor = M['m00']

            if divisor != 0.0:
                centroid_x = int(M['m10'] / divisor)  # Get the x-centriod the cnt
                centroid_y = int(M['m01'] / divisor)  # get the y-centriod the cnt

                XTrajectory.append(centroid_x)
                YTrajectory.append(centroid_y)
                CID.append(int(index2))
                frameID.append(int(0))
                period.append(int(elipse))

            for ii, frame in enumerate(f[1:]):
                ii += 1
                distances = []

                NextFrame, _ = watershedsegmentation(frame, fNum, ii)
                print("CIN :{}".format(index2))
                print ("FID: {}".format(ii))

                for indices, NextFrameCell in enumerate(NextFrame):
                    ref = cv2.moments(NextFrameCell)
                    denominator = ref['m00']

                    if denominator != 0.0:
                        centroid_xx = int(ref['m10'] / denominator)  # Get the x-centriod the cnt
                        centroid_yy = int(ref['m01'] / denominator)  # get the y-centriod the cnt
                        # compute the sum squared difference of the centriods
                        sim = np.sqrt((centroid_xx - centroid_x) ** 2 + (centroid_yy - centroid_y) ** 2)
                        distances.append(sim)
                        CloseCells.append(NextFrameCell)
                if distances is None:
                    continue

                MinDistance = min(distances)
                MinIndex = distances.index(MinDistance)
                indexedCell = CloseCells[MinIndex]

                M1 = cv2.moments(indexedCell)
                check = M1['m00']

                if check != 0.0:
                    centroid_xx = int(M1['m10'] / check)  # Get the x-centriod of the cnt
                    centroid_yy = int(M1['m01'] / check)  # get the y-centriod of the cnt

                # to make sure the movement of a cell is not beyond the reasonable velocity
                diff = abs(centroid_xx - centroid_x)
                diff2 = abs(centroid_yy - centroid_y)
                print('Moved {} pixel(s) on the x-axis'.format(diff))
                print('Moved {} pixel(s) on the y-axis'.format(diff2))

                if diff > 10:
                    continue

               
                elipse += 8
                # keep the trajectories of each cell
                XTrajectory.append(centroid_xx)
                YTrajectory.append(centroid_yy)
                period.append(int(elipse))
                frameID.append(int(ii))
                CID.append(index2)
                # update the previous centroid with the new centroid 
                centroid_x = centroid_xx
                centroid_y = centroid_yy

                centroid_xx = None
                centroid_yy = None

        centroid.append([frameID, CID, XTrajectory, YTrajectory, period])


        xx = centroid[0]
        CellCentroid.append(xx)
    cc = CellCentroid[0]
  
    unpacked = zip(cc[0], cc[1], cc[2], cc[3], cc[4])
    # save data to csv file format
    with open('/dirLast/data_' + str(count) + '.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'CellID', 'x-axis', "y-axis", 'time',))
        for value in unpacked:
            writer.writerow(value)









