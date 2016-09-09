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
from time import sleep
import random
from collections import deque
import sys
from skimage.color.colorlabel import label2rgb
import pylab as py

COLORS = [(139, 0, 0),
          (0, 100, 0),
          (0, 0, 139)]
def random_color():
    return random.choice(COLORS)

def VideoFrameReaders(VideoDirectory):
    cap = cv2.VideoCapture(VideoDirectory)
    timestamp = []
    frames = []
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            t1 = cap.get(0) #get the frame in seconds
            timestamp.append(t1)

            if frame == None:
                break;
            frames.append(frame)
    except EOFError:
      pass
    return timestamp,frames

def watershed_seg(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    contors = []
    try:
        image = frame
        fgmask = fgbg.apply(image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
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

        mask = np.zeros(gray.shape, dtype="uint8")
        for label in np.unique(labels):
             # if the label is zero, we are examining the 'background'
              #so simply ignore it
             if label == 0:
                 continue

            #otherwise, allocate memory for the label region and draw
             # it on the mask
             mask[labels == label] = 255
             cv2.imshow('masked', mask)
             # detect contours in the mask and grab the largest one
             cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)[-2]
             cnt = cnts[0]
             #areas = [cv2.contourArea(c) for c in cnts]
             #max_index = np.argmax(areas)
             #cnt = cnts[max_index]
             contors.append(cnt)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
             #   break


    except EOFError:
        pass
    return contors

### main function

if __name__=='__main__':
    # read frames of the video
    path = glob.glob('/run/user/1000/gvfs/smb-share:server=mscopy2.ugent.be,share=microscopy/CM/CM_P013_immune_cells/CM_P013_E016/CM_P013_E016_raw/CM_P013_E016_microscope/25-04-2013/movies/*.avi')
    print path
    centroid = []
   # path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');
    orgCont = []
    for i in range(len(path)):
        t, f = VideoFrameReaders(path[i])

        min_dist = sys.maxint
        org_cnts = watershed_seg(f[0])
        closet_cont = []
        for i,frame in enumerate(f[1:]):

            ref_cnts = watershed_seg(frame)
            conter1 = 0

            for i2,org_conts in enumerate(org_cnts):
                print i2
                # mark the contour in the firt frame
                ((x, y), r) = cv2.minEnclosingCircle(org_conts)
                cv2.circle(f[0], (int(x), int(y)), int(r), random_color(), 0)
                cv2.putText(f[0], "*{}".format(i2), (int(x) , int(y)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, random_color(), 2)

                org_m = np.asarray(org_conts)
                M = cv2.moments(org_conts)
                centroid_x = int(M['m10']/M['m00'])  # Get the x-centriod the cnt
                centroid_y = int(M['m01']/M['m00'])  # get the y-centriod the cnt
                centroid.append([centroid_x,centroid_y,t[i]])
                print centroid

                cv2.circle(f[0], (centroid_x, centroid_y), 3, random_color(), -1)

                # show the output image
                cv2.imshow("F{}".format(i2), f[0])
                #cv2.imshow("Referenced", frame)
                conter2 = 0
                dist,y_list,x_list, y_list2 = [],[],[],[]
                print("length:{}".format(len(ref_cnts)))
                for ref_conts  in ref_cnts:

                    try:

                        orgCont.append(ref_conts)
                        M1 = cv2.moments(ref_conts)
                        centroid_xx = int(M1['m10'] / M1['m00'])  # Get the x-centriod the cnt
                        centroid_yy = int(M1['m01'] / M1['m00'])  # get the y-centriod the cnt
                      #  cv2.circle(frame, (centroid_xx, centroid_yy), 3, random_color(), -1)


                        ret = cv2.matchShapes(org_conts,ref_conts,1,0.0)
                        print ('Distance %f between cont %d and cont %d, '
                               'at time %d seconds'
                               % (ret,conter1, conter2, t[i+1]))
                        conter2 += 1

                        if ret < 0.3:
                            min_dist = ret
                            dist.append(ret)
                            closet_cont.append(ref_conts)
                    except Exception:
                        pass
                m = min(i for i in dist if i >0 )
                index = dist.index(m)
                print index

                closet_cont2 =  closet_cont[2]
                ax1 = py.subplot(111)
                ax1.plot(closet_cont2, "k")
                print closet_cont2


                #print orgCont[index]
                M1 = cv2.moments(orgCont[2])
                centroid_xx = int(M1['m10'] / M1['m00'])  # Get the x-centriod the cnt
                centroid_yy = int(M1['m01'] / M1['m00'])  # get the y-centriod the cnt
                cv2.circle(frame, (centroid_xx, centroid_yy), 3, 255, -1)

                cv2.drawContours(frame, closet_cont2,-1 , random_color(), 3)
                cv2.imshow('Output', frame)

                cv2.waitKey(0)
                cv2.destroyAllWindows()


            sleep(2)
            exit()
            conter1 += 1
            count += 1






