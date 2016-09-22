# menu.py
import Tkinter as tk
from Tkinter import *
import tkMessageBox
import pygubu
from collections import deque
import zipfile
try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2
import pygubu

from Tkinter import *
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from tkFileDialog import asksaveasfilename
from tkFileDialog import asksaveasfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkMessageBox
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from sklearn import neighbors
import os, csv
import cv2
import tkFont
from PIL import Image as img2
from Tkinter import Image
import numpy as np
from libtiff import TIFF
import platform
import time
from Tkinter import tkinter
import ttk
import gifmaker
import ImageSequence
import glob
import ImageTk
import mahotas

# get the platform
if platform.system() == 'Linux':
    dash = '/'
if platform.system() == 'Windows':
    dash = ''

tmppath = os.getcwd()
tmppath = tmppath + dash + time.strftime("%d_%m_%Y_%I")

if os.path.exists(tmppath) is False:
    os.mkdir(tmppath)

# create directories

trackingdir = os.path.join(tmppath + os.sep, 'tracking/')
trajectorydir = os.path.join(tmppath + os.sep, 'finalplot/')
overlaytrajectorydir = os.path.join(tmppath + os.sep, 'overlaytrajecory/')
overlaytrajectoryanidir = os.path.join(tmppath + os.sep, 'overlaytrajecoryani/')
masktrajectorydir = os.path.join(tmppath + os.sep, 'masktrajector/')
csvdir = os.path.join(tmppath + os.sep, 'datafiles/')

if os.path.exists(trackingdir) is False:
    os.mkdir(trackingdir)

if os.path.exists(trajectorydir) is False:
    os.mkdir(trajectorydir)

if os.path.exists(csvdir) is False:
    os.mkdir(csvdir)

if os.path.exists(overlaytrajectorydir) is False:
    os.mkdir(overlaytrajectorydir)

if os.path.exists(masktrajectorydir) is False:
    os.mkdir(masktrajectorydir)

if os.path.exists(overlaytrajectoryanidir) is False:
    os.mkdir(overlaytrajectoryanidir)

# parameters for Shi-Tomasi Corner detectors


# Parameters for lucas kanade optical flow

lk_params = dict(winSize=(20, 20), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

# Create some random colors

color = np.random.randint(0, 255, (500, 3))


def read_tiff(self, path):
    start = time.time()
    frames = []
    tif = TIFF.open(path, mode='r')
    try:
        for cc, tframe in enumerate(tif.iter_images()):
            frames.append(tframe)
            print cc
            cv2.imwrite('/home/sami/framesPPT/nextFrame/newImg/%d.jpg'%cc, tframe)
    except EOFError:
        pass
    end = start - time.time()
    print end
    return frames


def read_avi(self, path):
    frames, timestamp = [], []
    cap = cv2.VideoCapture(path)
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
    return frames, timestamp


def read_others(self, path):
    frames = cv2.imread(path)
    return frames


def resize_image(trackingdir):
    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def animate(self, trackingdir):
    # look_directory(trackingdir)

    # CHECK IF LIST IS EMPTY
    if len(self.gifBackgroundImages) == 0:

        # CREATE FILES IN LIST
        for foldername in os.listdir(trackingdir):
            self.gifBackgroundImages.append(foldername)

        self.gifBackgroundImages.sort(key=lambda x: int(x.split('.')[0]))

    if self.atualGifBackgroundImage == len(self.gifBackgroundImages):
        self.atualGifBackgroundImage = 0
    try:
        self.background["file"] = trackingdir + self.gifBackgroundImages[self.atualGifBackgroundImage]
        self.label1["image"] = self.background
        self.atualGifBackgroundImage += 1
    except EOFError:
        print (trackingdir + self.gifBackgroundImages[self.atualGifBackgroundImage])
        pass

    # MILISECONDS\/ PER FRAME
    self.after(300, lambda: animate(self, trackingdir))


def morph_dilate(self, image):
    image = cv2.dilate(image, cv2.MORPH_DILATE, kernel)
    return image


def morph_close(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def morph_open(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def morph_gradient(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return image


def morph_erode(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
    return image


def white_background(image, kernel):

    im = cv2.threshold(image, 173, 255, cv2.THRESH_BINARY)
    im = im[1]
    dilation = cv2.dilate(im, kernel, iterations=1)
    gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)


    shifted = cv2.pyrMeanShiftFiltering(closing, 10, 20)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    # create a mask
    mask2 = np.zeros(image.shape, dtype="uint8")

    #  loop over the unique labels returned by the Watershed  algorithm for
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask2[labels == label] = 255
        # close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    return mask2


def basic_seg(img, images):
    noOfFrames = len(images)
    bgFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1, 4):
        bgFrame = bgFrame / 2 + \
                  cv2.cvtColor((images[i]),
                               cv2.COLOR_BGR2GRAY) / 2

    # Array to save the object locations
    objLocs = np.array([None, None])

    # Kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Display the frames like a video
    # Read each frame
    frame = images[1]

    # Perform background subtraction after median filter
    diffFrame = cv2.absdiff(cv2.cvtColor(cv2.medianBlur(frame, 7), \
                                         cv2.COLOR_BGR2GRAY), cv2.medianBlur(bgFrame, 7))

    # Otsu thresholding to create the binary image
    [th, bwFrame] = cv2.threshold(diffFrame, 0, 255, cv2.THRESH_OTSU)

    # Morphological opening operation to remove small blobs
    bwFrame = cv2.morphologyEx(bwFrame, cv2.MORPH_OPEN, kernel)

    return bwFrame

def black_background(image, kernel):

    shifted = cv2.pyrMeanShiftFiltering(image, 10, 39)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # create a mask
    mask2 = np.zeros(gray.shape, dtype="uint8")
    #  loop over the unique labels returned by the Watershed  algorithm for
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask2[labels == label] = 255
    return mask2


# histgram equalization
def histogram_equaliz(image):
    old_gray_image = cv2.equalizeHist(image)
    return old_gray_image


def shi_tomasi(image, maxCorner, qualityLevel, MinDistance):
    # detect corners in the image

    corners = cv2.goodFeaturesToTrack(image,
                                      maxCorner,
                                      qualityLevel,
                                      MinDistance,
                                      mask=None,
                                      blockSize=7)

    return corners


def harris_corner(image, maxCorner, qualityLevel, minDistance):
    corners = cv2.goodFeaturesToTrack(image,  # img
                                      maxCorner,  # maxCorners
                                      qualityLevel,  # qualityLevel
                                      minDistance,  # minDistance
                                      None,  # corners,
                                      None,  # mask,
                                      7,  # blockSize,
                                      useHarrisDetector=True,  # useHarrisDetector,
                                      k=0.05  # k
                                      )
    return corners


def optical_flow(self, frames, old_gray_image1, intialPoints, segMeth, maxCorner, qualityLevel, minDistance,
                 updateconvax, progessbar):

    old_gray_image2 = cv2.cvtColor(old_gray_image1, cv2.COLOR_BGR2GRAY)
    old_gray_image2 = histogram_equaliz(old_gray_image2)


    print intialPoints

    mask = np.zeros_like(old_gray_image1,)

    finalFrame = len(frames)

    trajectoriesX, trajectoriesY, cellIDs, frameID = [], [], [], []

    for i, frame in enumerate(frames):
        try:
            new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # perform histogram equalization to balance image color intensity
            new_gray_image = histogram_equaliz(new_gray_image)

            progessbar.step(i*2)


            newPoints, st, err = cv2.calcOpticalFlowPyrLK(old_gray_image2, new_gray_image, intialPoints, None,
                                                          **lk_params)

            # Select good points

            good_new = newPoints[st == 1]

            good_old = intialPoints[st == 1]

            # draw the tracks
            cont = len(good_new)
            initialValue = 0
            for ii, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()

                c, d = old.ravel()

                #mask = cv2.line(mask, (a, b), (c, d), color[ii].tolist(), 2)
                mask = cv2.line(mask, (a, b), (c, d), (255, 255, 255), 2)

                #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                #frame = cv2.putText(frame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #                        color[ii].tolist())

                #nextFrame = cv2.putText(nextFrame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #                                                (255, 255, 255))

                img = cv2.add(frame, mask)

                edged2 = np.hstack([mask, img])

                # Now update the previous frame and previous points
                old_gray_image2 = new_gray_image.copy()

                intialPoints = good_new.reshape(-1, 1, 2)
                # Keep the data of for later processing
                trajectoriesX.append(a)
                trajectoriesY.append(b)
                cellIDs.append(ii)
                frameID.append(i)
            r = 500.0 / img.shape[1]
            dim = (500, int(img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave(trackingdir + '%d.gif' % i, img)
            mahotas.imsave(masktrajectorydir + '%d.gif' % i, mask)
            mahotas.imsave(overlaytrajectorydir + '%d.gif' % i, resized)

            tmp_path = trackingdir + '%d.gif' % i
            image = img2.open(str(tmp_path))
            image = ImageTk.PhotoImage(image)
            root.image = image
            imagesprite = updateconvax.create_image(283, 187, image=image, anchor='c')
            time.sleep(1)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == finalFrame - 1:
                cv2.imwrite(trajectorydir + 'finalTrajectory.png', img)
                cv2.imwrite(trajectorydir + 'Plottrajector.png', mask)
                image = img2.open(str(tmp_path))
                image = ImageTk.PhotoImage(image)
                root.image = image
                imagesprite = updateconvax.create_image(280, 193, image=image, anchor='c')
        except EOFError:
            continue


    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY)
    with open(csvdir + 'data.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'CellIDs', 'x-axis', "y-axis",))
        for value in unpacked:
            writer.writerow(value)


def shape_matching(self, newframes, oldframes, kernel, segMeth=''):
    # PERFORM SEGMENTATION USING WATERSHED ALGORITHM
    if trackingMethd == 'white':
        mask, _ = white_background(oldframes)
    if trackingMethd == 'black':
        mask, _ = black_background(oldframes)
    else:
        tkMessageBox.showinfo('Optical flow', 'defualt tracking is set to optical flow')


def centroid_matching(self, oldframe, frames, kernel, intialPoints,  maxCorner, qualityLevel, minDistance,
                 updateconvax, segMeth):
    # PERFORM SEGMENTATION USING WATERSHED ALGORITHM
    if segMeth == 'white':
        mask = white_background(oldframe)
    if segMeth == 'black':
        mask = black_background(oldframe)
    if segMeth == 'haris':
        #intialPoints = shi_tomasi(oldframe, maxCorner, qualityLevel, minDistance)
        print 'ok'
    if segMeth == 'shi':
        #intialPoints = harris_corner(oldframe, maxCorner, qualityLevel, minDistance)
        print 'ok'
    print segMeth
    mask =  mask = np.zeros_like(oldframe,)
    for i,frame in enumerate(frames):
        #try:
            tm_img =   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tm_img = histogram_equaliz(tm_img)
            if segMeth =='shi':
                newPoints = shi_tomasi(tm_img, maxCorner, qualityLevel, minDistance)
            if segMeth == 'haris':
                newPoints == harris_corner(tm_img, maxCorner,qualityLevel, minDistance)

            good_new = newPoints

            good_old = intialPoints

            for ii, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()

                c, d = old.ravel()

                mask = cv2.line(mask, (a, b), (c, d), color[ii].tolist(), 2)
                #mask = cv2.line(mask, (a, b), (c, d), (255, 255, 255), 2)

                #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                #frame = cv2.putText(frame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #                        color[ii].tolist())

                #frame = cv2.putText(frame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #                        (255, 255, 255))

                img = cv2.add(frame, mask)

                edged2 = np.hstack([mask, img])

                # Now update the previous frame and previous points
                oldframe = frame.copy()

                intialPoints = good_new.reshape(-1, 1, 2)

                # Keep the data of for later processing
                #trajectoriesX.append(a)
                #trajectoriesY.append(b)
                #cellIDs.append(ii)
                #frameID.append(i)
            r = 500.0 / img.shape[1]
            dim = (500, int(img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave(trackingdir + '%d.gif' % i, img)
            mahotas.imsave(masktrajectorydir + '%d.gif' % i, mask)
            mahotas.imsave(overlaytrajectorydir + '%d.gif' % i, resized)

            tmp_path = trackingdir + '%d.gif' % i
            image = img2.open(str(tmp_path))
            image = ImageTk.PhotoImage(image)
            root.image = image
            imagesprite = updateconvax.create_image(280, 193, image=image, anchor='c')
            time.sleep(1)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == 100 - 1:
                cv2.imwrite(trajectorydir + 'finalTrajectory.png', img)
                cv2.imwrite(trajectorydir + 'Plottrajector.png', mask)
                image = img2.open(str(tmp_path))
                image = ImageTk.PhotoImage(image)
                root.image = image
                imagesprite4 = updateconvax.create_image(280, 193, image=image, anchor='c')

            #except EOFError:
            #    continue

    #unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY)
    #with open(csvdir + 'data.csv', 'wt') as f1:
    #    writer = csv.writer(f1, lineterminator='\n')
    #    writer.writerow(('frameID', 'CellIDs', 'x-axis', "y-axis",))
    #    for value in unpacked:
    #        writer.writerow(value)

'''
def centroid_matching(self, newframes, oldframes, kernel, intialPoints,  maxCorner, qualityLevel, minDistance,
                 updateconvax, segMeth=''):
    # PERFORM SEGMENTATION USING WATERSHED ALGORITHM
    if segMeth == 'white':
        mask = white_background(oldframes)
    if segMeth == 'black':
        mask = black_background(oldframes)
    if segMeth =='haris':
        image = shi_tomasi(newframes,maxCorner, qualityLevel, minDistance)
    if segMeth =='shi':
        image = harris_corner(newframes,maxCorner, qualityLevel, minDistance)

    else:
        tkMessageBox.showinfo('Optical flow', 'defualt tracking is set to optical flow')

    # perform tracking
    for index, refCont in enumerate(mask):

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

            if trackingMethd == 'white':
                NextFrame = white_background(frame)
            if trackingMethd == 'black':
                NextFrame = black_background(frame)

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
                AbsDifference = abs(previous_xx - centroid_xx)
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

                # keep the trajectories of each cell

                refCont = NextFrameCell
                XTrajectory.append(centroid_xx)
                YTrajectory.append(centroid_yy)
                period.append(t[i])
                frameID.append(int(i))
                CID.append(int(index))
            # free the distances
            distances = []
            centroid_xx = None
            centroid_yy = None
            del distances

        centroid.append([frameID, CID, XTrajectory, YTrajectory, period])
        xx = centroid[0]

        # free variables

        del previous_yy

        del previous_xx

    cellCentroid.append(xx)
    cc = cellCentroid[0]

    unpacked = zip(cc[0], cc[1], cc[2], cc[3], cc[4])

    with open('Cancer_shape6.1_' + str(count) + '.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'CellID', 'x-axis', "y-axis", 'time',))
        for value in unpacked:
            writer.writerow(value)
    count += 1
'''

class Application:
    def __init__(self, master):

        # 1: Create a builder

        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('celltracker.ui')

        # 3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)

        # 4: Get the labeled frame
        self.labelframe1 = builder.get_object("Labelframe_19")

        # 5:  Get the filename or path
        self.pathchooserinput_3 = builder.get_object("pathchooserinput_3")

        # 6: Read the files
        self.button = builder.get_object("Button_10")

        # 7: Create a progress bar
        self.progressdialog = ttk.Progressbar(self.labelframe1, mode='indeterminate', value=0)
        self.progressdialog.grid(row=2, column=0, sticky=N + E + W)

        self.Labelframe_22 = builder.get_object("Labelframe_22")
        self.progressdialog2 = ttk.Progressbar(self.Labelframe_22, mode='indeterminate', value=0)
        self.progressdialog2.grid(row=3, column=0, columnspan=5,sticky=N + E + W)

        # 8:  Manage a segmentation parameters

        self.labelframe2 = builder.get_object("Labelframe_12")

        # 8.1: Scale label
        self.label = Label(self.labelframe2)
        self.label.grid(row=1, column=5, sticky=W)
        self.fixscale = 0.5
        self.label.configure(text=self.fixscale)

        # 8.2: Entry
        self.cellEstimate = 200
        self.minDistance = 10

        # 9: Perform segmentation

        self.preview = builder.get_object("Button_1")

        self.convax1 = builder.get_object("Canvas_4")

        self.segmentation = self.builder.get_variable("seg")
        self.color = self.builder.get_variable("background")

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 10: Create a tracking labels

        self.track = builder.get_variable("track")

        self.trackconvax = builder.get_object("Canvas_2")

        #11: Create a file mining

        self.generatefile = builder.get_object("Button_3")

        #12: about the toolbox

        self.clear = builder.get_object("Button_11")

        builder.connect_callbacks(self)

        ##: set global variable
        self.frames, self.timestamp = [], []

    def readfile_process_on_click(self):
        # Get the path choosed by the user

        path = self.pathchooserinput_3.cget('path')

        # show the path
        if path:
            tkMessageBox.showinfo('You choosed', str(path))

            # Set a global variable
            # Check for the file format
            if '.tif' in path:
                tif = TIFF.open(path, mode='r')
                try:
                    for cc, tframe in enumerate(tif.iter_images()):
                        self.frames.append(tframe)
                        self.progressdialog.step(cc)
                        self.progressdialog.update()


                        # self.mainwindow.update()
                except EOFError:
                    tkMessageBox.showinfo('Error', 'file cant be read!!!')
                    pass
                self.progressdialog.stop()

            if '.avi' in path:

                cap = cv2.VideoCapture(path)
                cc = 0
                try:
                    while cap.isOpened():
                        ret, img = cap.read()
                        # get the frame in seconds
                        t1 = cap.get(0)
                        self.timestamp.append(t1)
                        if img is None:
                            break
                        self.frames.append(img)
                        self.progressdialog.step(cc)
                        self.progressdialog.update()
                        time.sleep(0.1)
                        cc += 1
                        # self.mainwindow.update()
                except EOFError:
                    tkMessageBox.showinfo('Error', 'file cant be read!!!')
                    pass
                self.progressdialog.stop()
            tmp_img = self.frames[0]
            r = 500.0 / tmp_img.shape[1]
            dim = (500, int(tmp_img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(tmp_img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave('raw_image.gif', resized)
            image1 = img2.open('raw_image.gif')

            image1 = ImageTk.PhotoImage(image1)
            root.image1 = image1
            _ = self.convax1.create_image(270, 155, image=image1, anchor='c')

        else:
            tkMessageBox.showinfo("No file", "Choose a file to process")

    # segmentation preview
    def previe_on_click(self):
        "Display the values of the 2 x Entry widget variables"
        self.cellEstimate = self.builder.get_object('Entry_1')
        self.minDistance = self.builder.get_object('Entry_3')
        self.preconvax = self.builder.get_object("Canvas_5")

        self.cellEstimate = self.cellEstimate.get()
        self.minDistance = self.minDistance.get()

        if self.frames:
            # normalize histogram for improving the image contrast
            self.normalizedImage = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
            self.normalizedImage = histogram_equaliz(self.normalizedImage)
            self.segMethod = self.segmentation.get()

            if self.segMethod == 2:
                if self.color.get() == 1:
                    self.prev_image = black_background(self.frames[0], self.kernel)
                    r = 500.0 / self.prev_image.shape[1]
                    dim = (500, int(self.prev_image.shape[0] * r))

                    # perform the actual resizing of the image and show it
                    self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                    mahotas.imsave('SegImage.gif', self.prev_image)
                    tmp_pre = img2.open('SegImage.gif')
                    tmp_pre = ImageTk.PhotoImage(tmp_pre)
                    root.tmp_pre = tmp_pre
                    segprev = self.preconvax.create_image(280, 185, image=tmp_pre)
                if self.color.get() == 2:
                    self.prev_image = white_background(self.frames[0], self.kernel)
                    r = 500.0 / self.prev_image.shape[1]
                    dim = (500, int(self.prev_image.shape[0] * r))

                    # perform the actual resizing of the image and show it
                    self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                    mahotas.imsave('SegImage.gif', self.prev_image)
                    tmp_pre = img2.open('SegImage.gif')
                    tmp_pre = ImageTk.PhotoImage(tmp_pre)
                    root.tmp_pre = tmp_pre
                    segprev = self.preconvax.create_image(280, 185, image=tmp_pre)
            if self.segMethod == 3:
                self.prev_image = harris_corner(self.normalizedImage, int(self.cellEstimate), float(self.fixscale),
                                                int(self.minDistance))
                for corner in self.prev_image:
                    x, y = corner[0]
                    cv2.circle(self.normalizedImage, (int(x),int(y)),5, (0,0,255),-1)

                r = 500.0 / self.normalizedImage.shape[1]
                dim = (500, int(self.normalizedImage.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.normalizedImage = cv2.resize(self.normalizedImage, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.normalizedImage)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(280, 185, image=tmp_pre)

            if self.segMethod == 4:
                self.prev_image = shi_tomasi(self.normalizedImage, int(self.cellEstimate), float(self.fixscale),
                                             int(self.minDistance))
                for corner in self.prev_image:
                    x, y = corner[0]
                    cv2.circle(self.normalizedImage, (x, y), 5, (0, 255, 0), -1)

                r = 500.0 / self.normalizedImage.shape[1]
                dim = (500, int(self.normalizedImage.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.normalizedImage = cv2.resize(self.normalizedImage, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.normalizedImage)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(280, 185, image=tmp_pre)

            if self.segMethod == 5:
                self.prev_image = basic_seg(self.frames[0], self.frames)
                #print corners
                '''for corner in corners:
                    x, y = corner
                    cv2.circle(self.normalizedImage, (int(x), int(y)), 5, (0, 0, 255), -1)'''



                r = 500.0 / self.prev_image.shape[1]
                dim = (500, int(self.prev_image.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.prev_image)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(280, 185, image=tmp_pre)
        else:
            tkMessageBox.showinfo('No file', 'no data is found!!!')

        return self.segMethod

    # perform tracking
    def track_on_click(self):
        if self.frames:
            self.cellEstimate = self.builder.get_object('Entry_1')
            self.minDistance = self.builder.get_object('Entry_3')

            if self.frames:
                self.normalizedImage = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
                self.normalizedImage = histogram_equaliz(self.normalizedImage)
            else:
                pass


            if self.segmentation.get() == 2:
                if self.color.get() == 1:
                    self.mask = black_background(self.frames[0], self.kernel)
                    #self.mask = histogram_equaliz(self.mask)
                    self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                    float(self.fixscale),
                                                    int(self.minDistance.get()))
                    self.seg = 'black'
                if self.color.get() == 2:
                    self.mask = white_background(self.frames[0], self.kernel)
                    self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                    self.mask = histogram_equaliz(self.mask)

                    self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                    float(self.fixscale),
                                                    int(self.minDistance.get()))
                    self.seg = 'white'
            if self.segmentation.get() == 3:
                self.initialpoints = harris_corner(self.normalizedImage, int(self.cellEstimate.get()), float(self.fixscale),
                                                   int(self.minDistance.get()))
                self.seg = 'haris'
            if self.segmentation.get() == 4:
                self.initialpoints = shi_tomasi(self.normalizedImage, int(self.cellEstimate.get()), float(self.fixscale),
                                                int(self.minDistance.get()))
                self.seg = 'shi'

            if self.segmentation.get() == 5:
                self.mask =  basic_seg(self.frames[0], self.frames)
                self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()), float(self.fixscale),
                                                int(self.minDistance.get()))


                self.seg = 'basic'

            # manipulate a tracking method

            if self.track.get() == 8:
                tkMessageBox.showinfo('..','Segmentation method: %s \n' %self.seg, )

                optical_flow(self, self.frames[1:], self.frames[0], self.initialpoints, str(self.seg),
                             int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()), self.trackconvax, self.progressdialog2)

                #for i in range(5):
                #    for j in range(4):
                #        l = Label(text='%d.%d' % (i, j), relief=RIDGE)
                #        l.grid(row=i, column=j, sticky=NSEW)
            if self.track.get() == 5:
                centroid_matching(self,self.frames[0], self.frames[1:],self.kernel,self.initialpoints,int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()), self.trackconvax,
                                 self.seg)
        else:
            tkMessageBox.showinfo('Missing data', 'no data to process')

    # scale
    def on_scale_click(self, event):

        scale = self.builder.get_object('Scale_1')
        self.fixscale = float("%.1f" % round(scale.get(), 1))
        self.label.configure(text=str(self.fixscale))


    # generate files
    def generate_click(self):

        if not (os.path.join(overlaytrajectoryanidir,'animation.gif')):
            save_gif = True
            title = ''
            images, imgs = [], []
            for foldername in os.listdir(overlaytrajectorydir):
                images.append(foldername)
            images.sort(key=lambda x: int(x.split('.')[0]))

            for _, file in enumerate(images):
                print file
                print os.path.join(overlaytrajectorydir,file)
                im = img2.open(os.path.join(overlaytrajectorydir,file))

                imgs.append(im)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

            im_ani = animation.ArtistAnimation(fig, ims, interval=600, repeat_delay=500, blit=False)

            if save_gif:
                im_ani.save(overlaytrajectoryanidir + 'animation.gif', writer='imagemagick')

        self.file_opt = options = {}
        options['filetypes'] = [('gif', '.gif'), ('text files', '.tif')]
        options['initialfile'] = 'animation.gif'
        options['parent'] = overlaytrajectoryanidir
        self.savefile()

    def savefile(self):
        print self.file_opt
        return asksaveasfile(mode='w', **self.file_opt)





    def save_as_zip(self):

        zf = zipfile.ZipFile("data.zip", "w")
        for dirname, subdirs, files in os.walk(tmppath):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

    def clear_frame(self, master):

        for child in self.mainwindow.winfo_children():
            child.destroy()
        self.mainwindow.pack_forget()
        self.mainwindow.grid_forget()

        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('celltracker.ui')

        # 3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)




if __name__ == '__main__':
    root = tk.Tk()
    menu = Menu(root)
    root.config(menu=menu)

    file = Menu(menu)
    # file.add_command(label='Open', command='')
    file.add_command(label='Exit', command=lambda: exit())

    menu.add_cascade(label='File', menu=file)

    edit = Menu(menu)

    # adds a command to the menu option, calling it exit, and the
    # command it runs on event is client_exit
    edit.add_command(label="Undo")

    # added "file" to our menu
    menu.add_command(label="Open", command='')
    menu.add_command(label="Save", command='')
    menu.add_separator()
    menu.add_command(label="Exit", command=lambda: exit())
    menu.add_cascade(label="Edit", menu=edit)

    # root.geometry('1500x1000')

    img = Image("photo", file="/home/sami/Desktop/multimot-logo-e1437119906276.png")
    root.tk.call('wm', 'iconphoto', root._w, img)
    root.title("Cell Tracker")
    app = Application(root)
    root.mainloop()
