# this functions gets a frame by frame from an a video and an initially perform background substraction
#then perform basic morphological operations to help cleansing the images by filling holes and removing
# some whites. It also applies a mask of 3x3 on the images for directional change

#load  packages
import cv2
import os
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as ptl
import Image
import glob
from time import sleep
#import cancer_cell
from skimage import measure

path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');

ptl.close('all')

def videoframereaders(videodirectory):

    cap = cv2.VideoCapture(videodirectory)
    # define a kernel and subtract the background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    timestamp = []
    count = 0
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            time = cap.get(0)
            timestamp.append(time)
            print timestamp

            if frame == None:
                break;
            image = frame
            fgmask = fgbg.apply(image)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            cv2.imshow('frame', fgmask)

            #take the image and perform pyramid mean shift filtering to aid the thresholding step
            fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
            shifted = cv2.pyrMeanShiftFiltering(fgmask, 10, 10)
            print shifted
            cv2.imshow("Input", image)
            gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            cv2.imshow("Thresh", thresh)
            L = measure.label(thresh)
            print "Number of components:", np.max(L)

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            print("[INFO] {} unique contours found".format(len(cnts)))

            # loop over the contours
            for (i, c) in enumerate(cnts):
                # draw the contour
                ((x, y), _) = cv2.minEnclosingCircle(c)
                #cv2.putText(image, "*{}".format(i + 1), (int(x) - 10, int(y)),
                 #   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 0)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 1)

                # show the output image
                cv2.imshow("Contour", image)

                #cv2.imshow('window-name',image)
                # cv2.imwrite("/home/sami/Desktop/movies/extractFrames/frame%d.jpg" % count, image)
            count = count + 1
            sleep(5)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    except EOFError:
        pass
    return count,timestamp,

print path

c,t = videoframereaders(path[1])


# get frames from a directory

frames = glob.glob('/home/sami/Desktop/movies/extractFrames/*.jpg')
