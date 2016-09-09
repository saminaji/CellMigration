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
import seaborn as sns
import random
import sys
import time
from skimage.color.colorlabel import label2rgb
import pylab as py
import csv
from IPython import display

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


def watershedsegmentation(frame):
    contors = []
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

        tt = 0
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it

            if label == 0:
                 continue

            #otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            #dilation = cv2.dilate(mask, kernel, iterations=2)
           # Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


            cv2.imshow('closed', mask)
            cv2.waitKey(0)
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)[-2]
            if tt > 0:
                break
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
                cv2.imshow('f', frame)
            except IndexError:
                continue
            tt += 1




            #cv2.drawContours(frame, cnt, -1, (255,0,255), 2)
            # cv2.imshow('f', frame)
            cv2.imwrite('Image_%d.png' % 1 , frame)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            # break

    except EOFError:
        pass
    return contors, mask

# main function

if __name__=='__main__':

    # read frames of the video
    path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')
    cellCentroid = []
    path = '/home/sami/9K5F98PS_F00000011.avi'

    # path = glob.glob('/home/sami/Desktop/movies/movies/*.avi')

    for p in range(len(path[0])):
        t, f = filereader(path)

        refFrame, mask2 = watershedsegmentation(f[0])

        frameID, period, CloseCells, XTrajectory, YTrajectory, CID = [], [], [], [], [], []

        for index, refCont in enumerate(refFrame):
            centroid = []
            count = 0
            M = cv2.moments(refCont)
            divisor = M['m00']

            if divisor != 0.0:
                centroid_x = int(M['m10'] / divisor)  # Get the x-centriod the cnt
                centroid_y = int(M['m01'] / divisor)  # get the y-centriod the cnt

                XTrajectory.append(centroid_x)
                YTrajectory.append(centroid_y)
                period.append(t[0])
                frameID.append(int(0))

            for i, frame in enumerate(f[1:]):
                i += 1
                distances = []

                NextFrame, _ = watershedsegmentation(frame)

                print("Number of Cells:{}".format(len(NextFrame)))
                print("CIN :{}".format(index))
                for index2, NextFrameCell in enumerate(NextFrame):
#                    a, b = new.ravel()  # tmp new value
                   #  c, d = old.ravel()  # tmp old value

                    # draws a line connecting the old point with the new point
#                    cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

                    ref = cv2.moments(NextFrameCell)
                    denominator = ref['m00']

                    if denominator != 0.0:
                        centroid_xx = int(ref['m10'] / denominator)  # Get the x-centriod the cnt
                        centroid_yy = int(ref['m01'] / denominator)  # get the y-centriod the cnt
                        # compute the absolute difference of the centriods
                        sim = np.sqrt((centroid_xx - centroid_x) ** 2 + (centroid_yy - centroid_y) ** 2)

                        distances.append(sim)
                        CloseCells.append(NextFrameCell)

                MinDistance = min(distances)
                MinIndex = distances.index(MinDistance)

                indexedCell = CloseCells[MinIndex]
                M1 = cv2.moments(indexedCell)
                check = M1['m00']

                if check != 0.0:
                    centroid_xx = int(M1['m10'] / check)  # Get the x-centriod the cnt
                    centroid_yy = int(M1['m01'] / check)  # get the y-centriod the cnt
                #print centroid_x, centroid_y
                # print centroid_xx, centroid_yy

                # to make sure the movement of a cell is  beyond the reasonable boundaries
                diff = abs(centroid_xx - centroid_x)
                diff2 = abs(centroid_yy - centroid_y)
                print('Moved {} pixel(s) on the x-axis'.format(diff))
                print('Moved {} pixel(s) on the y-axis'.format(diff2))

                if diff > 5:
                    continue

                # update the previous centroid with the new centroid of a particular cell
                # print centroid_xx, centroid_yy
                centroid_x = centroid_xx
                centroid_y = centroid_yy
                # keep the trajectories of each cell
                XTrajectory.append(centroid_xx)
                YTrajectory.append(centroid_yy)
                period.append(t[i])
                frameID.append(int(i))
                CID.append(int(index))

                print("Frame:{}".format(int(i)))

                '''plt.scatter(XTrajectory, YTrajectory)
                plt.plot(XTrajectory, YTrajectory, color='k', linestyle='-', linewidth=1)
                plt.show(False)
                plt.draw()
                time.sleep(1)
                plt.close(1)'''
                # display.display(plt.figure())
                # display.clear_output(wait=True)

                centroid_xx = None
                centroid_yy = None
                del distances

            centroid.append([frameID, CID, XTrajectory, YTrajectory, period])
            xx = centroid[0]

        cellCentroid.append(xx)
        cc = cellCentroid[0]

        unpacked = zip(cc[0], cc[1], cc[2], cc[3], cc[4])
        #  plt.plot(XTrajectory, YTrajectory, color='k', linestyle='-', linewidth=1)
        plt.scatter(cc[2], cc[3])
        #plt.xlim(0, 300)
        plt.ylim(0, 600)
        plt.xticks(np.linspace(0, 700, 1))
        plt.show(False)
        plt.draw()
        time.sleep(240)
        # Save figure in png  formats
        plt.savefig('trajectoriesCe.png', bbox_inches='tight')

        display.clear_output(wait=True)
        print frameID, CID

        with open('data_Centroid_tiff' + str(count) + '.csv', 'wt') as f1:
            writer = csv.writer(f1, lineterminator='\n')
            writer.writerow(('frameID', 'CellID', 'x-axis', "y-axis", 'time',))
            for value in unpacked:
                writer.writerow(value)
        count += 1








