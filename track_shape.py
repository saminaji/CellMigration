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


        ''' cv2.imshow('draw', fgmask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

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

        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            mask = np.zeros(gray.shape, dtype="uint8")
            if label == 0:
                 continue

            #otherwise, allocate memory for the label region and draw
            # it on the mask
            mask[labels == label] = 255

            #   detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnt = cnts[0]

            contors.append(cnt)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            # break

    except EOFError:
        pass
    return contors, mask



def mse(contA, contB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((contA.astype("float") - contB.astype("float")) ** 2)
    err /= float(contA.shape[0] * contA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# main function

if __name__=='__main__':

    # read frames of the video
    path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')

    cellCentroid = []
   # path = '/home/sami/9K5F98PS_F00000011.avi'
    path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'


    # path = glob.glob('/home/sami/Desktop/movies/movies/*.avi')

    for p in range(len(path[0])):
        t, f = filereader(path)

        for i in range(len(f)):


            #cv2.imwrite("raw_image.png", f[0])
            refFrame, _ = watershedsegmentation(f[0])
            #cv2.imwrite("processed_image.png", mask)

            frameID, period, ClosestCells, XTrajectory, YTrajectory, CID = [], [], [], [], [], []

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

                        # structural similarity index for the images
                        hd = cv2.createHausdorffDistanceExtractor()
                        d1 = hd.computeDistance(refCont, NextFrameCell)
                        if d1 is not None:
                            distances.append(d1)
                            ClosestCells.append(NextFrameCell)

                    # print distances

                    MinDistance = min(distances)
                    MinIndex = distances.index(MinDistance)

                    indexedCell = ClosestCells[MinIndex]
                    M1 = cv2.moments(indexedCell)
                    check = M1['m00']

                    if check != 0.0:
                        centroid_xx = int(M1['m10'] / check)  # Get the x-centriod the cnt
                        centroid_yy = int(M1['m01'] / check)  # get the y-centriod the cnt

                        # compared previous centroid
                        if i is 1:
                            previous_xx = centroid_xx
                            previous_yy = centroid_yy
                        AbsDifference = abs(previous_xx-centroid_xx)
                        AbsDifference2 = abs(previous_yy - centroid_yy)


                        # print AbsDifference
                        try:
                            while AbsDifference and AbsDifference2 > 3:

                                MinDistance = min(distances)
                                MinIndex = distances.index(MinDistance)

                                indexedCell = ClosestCells[MinIndex]
                                M1 = cv2.moments(indexedCell)
                                check = M1['m00']

                                if check != 0.0:
                                    centroid_xx = int(M1['m10'] / check)  # Get the x-centriod the cnt
                                    centroid_yy = int(M1['m01'] / check)  # get the y-centriod the cnt
                                AbsDifference = abs(previous_xx - centroid_xx)
                                AbsDifference2 = abs(previous_yy - centroid_yy)
                                if AbsDifference and AbsDifference2 <= 3:
                                    break
                                del distances[MinIndex]
                                del ClosestCells[MinIndex]
                        except ValueError:
                            pass
                        if centroid_xx > previous_xx + 5 or centroid_yy > previous_yy + 5:
                            continue
                        previous_xx = centroid_xx
                        print 'x', previous_xx
                        previous_yy = centroid_yy
                        print 'y', previous_yy

                        '''if index is 1:
                            print centroid_x, centroid_y
                            print XTrajectory, YTrajectory
                            exit()'''

                        # keep the trajectories of each cell
                        refCont = NextFrameCell
                        XTrajectory.append(centroid_xx)
                        YTrajectory.append(centroid_yy)
                        period.append(t[i])
                        frameID.append(int(i))
                        CID.append(int(index))

                    distances = []

                    print("Frame:{}".format(int(i)))

                    '''plt.scatter(XTrajectory, YTrajectory)
                    plt.plot(XTrajectory, YTrajectory, color='k', linestyle='-', linewidth=1)
                    plt.show(False)
                    plt.draw()
                    time.sleep(1)
                    plt.close(1)
                    # display.display(plt.figure())
                    # display.clear_output(wait=True)'''

                    centroid_xx = None
                    centroid_yy = None
                    del distances

                centroid.append([frameID, CID, XTrajectory, YTrajectory, period])
                xx = centroid[0]
                del previous_yy
                del previous_xx
                if index < 1:
                    break

            # print XTrajectory, YTrajectory

            cellCentroid.append(xx)
            cc = cellCentroid[0]


            unpacked = zip(cc[0], cc[1], cc[2], cc[3], cc[4])
            #  plt.plot(XTrajectory, YTrajectory, color='k', linestyle='-', linewidth=1)
            plt.scatter(cc[2], cc[3])
            #plt.xlim(0, 300)
            #plt.ylim(0,700)
            #plt.xticks(np.linspace(0, 700, 1))
            plt.show(False)
            plt.draw()
            time.sleep(240)
            # Save figure in png  formats
            plt.savefig('Shapetrajectories6.1.png', bbox_inches='tight')

            display.clear_output(wait=True)
            print frameID, CID

            with open('Cancer_shape6.1_' + str(count) + '.csv', 'wt') as f1:
                writer = csv.writer(f1, lineterminator='\n')
                writer.writerow(('frameID', 'CellID', 'x-axis', "y-axis", 'time',))
                for value in unpacked:
                    writer.writerow(value)
            count += 1








