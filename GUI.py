from Tkinter import *
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from tkFileDialog import asksaveasfilename
from tkFileDialog import asksaveasfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkMessageBox
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




# set the parameters of the shi-tomsi segmentation
v  = 100 # set the number of cells to be located
v1 = 0.5 # set the quality of the cells
v2 = 20 # minimum distance between each cell



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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# parameters for Shi-Tomasi Corner detectors

def shi_segm(number_of_cells, quality_of_cells, minimum_distance_between_cells):
    features = dict(maxCorners=number_of_cells,  # how many cells to locate
                    qualityLevel=quality_of_cells,  # quality level of the detected cells (0-1)
                    minDistance=minimum_distance_between_cells,  # minimum distance between each located cell
                    blockSize=7)  # search for block size

    return features


# Parameters for lucas kanade optical flow

lk_params = dict(winSize=(20, 20), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

# Create some random colors

color = np.random.randint(0, 255, (200, 3))


def read_tiff(self, path):
    start = time.time()
    frames = []
    tif = TIFF.open(path, mode='r')
    try:
        for cc, tframe in enumerate(tif.iter_images()):
            frames.append(tframe)
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


def white_background(image):
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
    mask2 = np.zeros(gray.shape, dtype="uint8")

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
    return mask


def black_background(image):
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

def shi_tomasi(image,feature_params ):
    # detect corners in the image
    old_gray_image = histogram_equaliz(image)
    corners = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)
    corners = np.float32(corners)

    # go through the corner
    for corner in corners:
        print corner

        x, y = corner[0]
        cv2.circle(old_gray_image,5, (x,y),255,-1)
    cv2.imshow('Best Corner', old_gray_image)
    cv2.waitKey(0)

    return corners



def optical_flow(self, frames, old_gray_image1,feature_params, segMeth ):
    old_gray_image2 = cv2.cvtColor(old_gray_image1, cv2.COLOR_BGR2GRAY)

    if segMeth == 'Lucas and Kanade':
        old_gray_image = histogram_equaliz(old_gray_image2)
    if segMeth == 'whiteBG':
        old_gray_image = white_background(old_gray_image2)
    if segMeth == 'blackBG':
        old_gray_image = black_background(old_gray_image2)

    intialPoints = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

    mask = np.zeros_like(old_gray_image1)

    finalFrame = len(frames)

    trajectoriesX, trajectoriesY, cellIDs, frameID = [], [], [], []

    for i, nextFrame in enumerate(frames):
        try:
            new_gray_image = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
            # perform histogram equalization to balance image color intensity
            if segMeth == 'Lucas and Kanade':
                new_gray_image = histogram_equaliz(new_gray_image)
            if segMeth == 'whiteBG':
                new_gray_image = white_background(new_gray_image)
            if segMeth == 'blackBG':
                new_gray_image = black_background(new_gray_image)

            # calculate optical flow

            newPoints, st, err = cv2.calcOpticalFlowPyrLK(old_gray_image, new_gray_image, intialPoints, None,
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
                mask = cv2.line(mask, (a, b), (c, d), (255,255,255), 2)


                # frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                #nextFrame = cv2.putText(nextFrame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #                        color[ii].tolist())

                nextFrame = cv2.putText(nextFrame, "%d" % ii, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (255,255,255))

                img = cv2.add(nextFrame, mask)

                edged2 = np.hstack([mask, img])

                # Now update the previous frame and previous points
                old_gray_image = new_gray_image.copy()

                intialPoints = good_new.reshape(-1, 1, 2)
                # Keep the data of for later processing
                trajectoriesX.append(a)
                trajectoriesY.append(b)
                cellIDs.append(ii)
                frameID.append(i)
            r = 600.0 / img.shape[1]
            dim = (600, int(img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(trackingdir + '%d.png' % i, img)
            cv2.imwrite(masktrajectorydir + '%d.png' % i, mask)
            cv2.imwrite(overlaytrajectorydir + '%d.png' % i, resized)
            # pb = ttk.Progressbar(root, orient="horizontal", length=finalFrame, mode="indeterminate")

            if i == finalFrame - 1:
                cv2.imwrite(trajectorydir + 'finalTrajectory.png', img)
                cv2.imwrite(trajectorydir + 'Plottrajector.png', mask)
        except EOFError:
            continue
        i * 2
        self.pbar_f.step(1)
        self.update()
        time.sleep(0.1)

    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY)
    with open(csvdir + 'data.csv', 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'CellIDs', 'x-axis', "y-axis",))
        for value in unpacked:
            writer.writerow(value)


def shape_matching(self, newframes, oldframes, kernel, trackingMethd=''):
    # PERFORM SEGMENTATION USING WATERSHED ALGORITHM
    if trackingMethd == 'white':
        mask, _ = white_background(oldframes)
    if trackingMethd == 'black':
        mask, _ = black_background(oldframes)
    else:
        tkMessageBox.showinfo('Optical flow', 'defualt tracking is set to optical flow')


def centroid_matching(self, newframes, oldframes, kernel, trackingMethd=''):
    # PERFORM SEGMENTATION USING WATERSHED ALGORITHM
    if trackingMethd == 'white':
        mask = white_background(oldframes)
    if trackingMethd == 'black':
        mask = black_background(oldframes)
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


class MyFrame(Frame):
    def __init__(self, master=None):

        Frame.__init__(self, master)

        self.grid()
        self.master.title("Cell Tracker")

        # set the main frame to a specific size
        self.master.minsize(width=1000, height=1000)
        self.master.minsize(width=1000, height=1000)
        self.master.columnconfigure(0, weight=1)
        # set title font style and size of the labels
        FontSize = tkFont.Font(family="Helvetica", size=10 )
        # set labels font styles and size
        FontSize2 = tkFont.Font(family="Helvetica", size=8)

        Frame1 = Frame(master, width=500, height=600, highlightbackground="azure2",
                       highlightthickness=3, bd=1)
        Frame1.grid(row=0, column=0, rowspan=3, columnspan=2, sticky=W)

        Frame2 = Frame(master, bg="yellow3", width=400, height=400, highlightbackground="azure2",
                       highlightcolor="green2", highlightthickness=3, bd=1)
        Frame2.grid(row=0, column=1, rowspan=1, columnspan=2, sticky=W)

        def enableEntry():
            self.SubmitButton.configure(state="normal")
            self.SubmitButton.update()

        def disableEntry():
            self.SubmitButton.configure(state="disabled")
            self.SubmitButton.update()
        self.l1 = LabelFrame(Frame1,text='Input data', width=10, height=50,  bg='orange1')
        #self.l1 = Label(Frame1, text='Input', width=30, height=1, font=FontSize, bg='light sea green')
        self.l1.grid(row=0, column=0,  sticky=S + W + E + N)

        self.l2 = Entry(self.l1,    width=30,  bg='white')
        self.l2.grid(row=1, column=0,  sticky=S + W)

        # Generate GIF file button
        #self.master.columnconfigure(0, weight=1)

        self.button1 = Button(self.l1, text="", command=self.load_file, width=5, height=1,
                              font=FontSize2, bg='white').grid(row=1, column=7, sticky= S + W)

        # set labels for parameters
        self.lF = LabelFrame(Frame1, text='Smoothing', width=10, height=50,  bg='orange1')
        self.lF.grid(row=0, column=1, sticky= S +  N )

        self.l2 = Radiobutton(   self.lF, text='Histogram equalization', width=19, height=1, font=FontSize2, bg='white')
        self.l2.grid(row=3, column=0, columnspan=2, sticky=S + W+ E+ N)

        # segmentation parameters
        # set labels for parameters
        self.l1 = Label(Frame1, text='Segmentation parameters', width=30, height=1, font=FontSize, bg='light sea green')
        self.l1.grid(row=4, column=0, columnspan=3, sticky=S + W + E + N)

        self.labelpar = Label(Frame1, text='Cell estimate', width=12, height=1, font=FontSize2, bg='white')
        self.labelpar.grid(row=5, column=0, columnspan=2, sticky=S + W)

        self.labelpar = Label(Frame1, text='Quality level', width=12, height=1, font=FontSize2, bg='white')
        self.labelpar.grid(row=5, column=1, columnspan=2, sticky=S + W)

        self.labelpar = Label(Frame1, text='Min distance', width=12, height=1, font=FontSize2, bg='white')
        self.labelpar.grid(row=5, column=2, columnspan=2, sticky=S + W)
        global v, v1, v2
        v = IntVar()
        e = Entry(Frame1, width=10, textvariable=v)
        e.grid(row=6, column=0, sticky=S + W)
        v.set("200")

        v1 = DoubleVar()
        e1 = Entry(Frame1, width=10, textvariable=v1)
        e1.grid(row=6, column=1, sticky=S + W)
        v1.set("0.5")

        v2 = DoubleVar()
        e2 = Entry(Frame1, width=10, textvariable=v2)
        e2.grid(row=6, column=2, sticky=S + W)
        v2.set("20")


        # set labels for algorithms
        self.l3 = Label(Frame1, text='Morphological operations', width=30, height=1, font=FontSize,
                        bg='light sea green')
        self.l3.grid(row=7, column=0, columnspan=3, sticky=S + W + E + N)

        # create and set the radio buttons for the choice of segmentation
        CheckVar1 = IntVar()
        CheckVar2 = IntVar()
        CheckVar3 = IntVar()
        CheckVar4 = IntVar()
        CheckVar5 = IntVar()
        CheckVar6 = IntVar()

        self.Check2 = Checkbutton(Frame1, text='deliation', variable=CheckVar2, width=13, height=1, font=FontSize2,
                                  bg='white', onvalue=1, offvalue=0)
        self.Check2.grid(row=8, column=0, columnspan=1, sticky=S + W)
        self.Check3 = Checkbutton(Frame1, text='closing', variable=CheckVar3, width=13, height=1, font=FontSize2,
                                  bg='white', onvalue=1, offvalue=0)
        self.Check3.grid(row=8, column=1, columnspan=1, sticky=S + W)

        self.Check4 = Checkbutton(Frame1, text='opening', variable=CheckVar4, width=13, height=1,
                                  font=FontSize2, bg='white', onvalue=1, offvalue=0)
        self.Check4.grid(row=8, column=2, columnspan=1, sticky=S + W)

        self.Check5 = Checkbutton(Frame1, text='erosion', variable=CheckVar5, width=13, height=1, font=FontSize2,
                                  bg='white', onvalue=1, offvalue=0)
        self.Check5.grid(row=9, column=0, columnspan=1, sticky=S + W)

        self.Check6 = Checkbutton(Frame1, text='gradient', variable=CheckVar6, width=13, height=1, font=FontSize2,
                                  bg='white', onvalue=1, offvalue=0)
        self.Check6.grid(row=9, column=1, columnspan=1, sticky=S + W)

        def dilatestate(*rgs):
            if CheckVar2.get():
                print "dilate me"

        def closestate(*args):
            if CheckVar3.get():
                print "close me"

        def openstate(*args):
            if CheckVar4.get():
                print "open me"

        def erodestate(*args):
            if CheckVar5.get():
                print "erode me"

        def gradientstate(*args):
            if CheckVar6.get():
                print "gradient me"

        CheckVar2.trace_variable("w", dilatestate)
        CheckVar3.trace_variable("w", closestate)
        CheckVar4.trace_variable("w", openstate)
        CheckVar5.trace_variable("w", erodestate)
        CheckVar6.trace_variable("w", gradientstate)

        # set labels for algorithms
        self.l1 = Label(Frame1, text='Segmentation algorithms', width=30, height=1, font=FontSize, bg='light sea green')
        self.l1.grid(row=10, column=0, columnspan=3, sticky=S + W + E + N)

        # create and functionate segmentation algorithms
        self.var = IntVar()
        self.var.set(0)  # initializing the choice, i.e. centroid
        varList = [('white-background', 0), ('black-background', 1), ('Shi-Tomasi', 2)]

        def ShowCHoice2():
            print self.var.get()

        for txt, val in varList:
            Radiobutton(Frame1,
                        text=txt,
                        padx=20,
                        variable=self.var,
                        command=ShowCHoice2,
                        value=val, width=13, height=1, font=FontSize2, bg='white',
                        activeforeground='blue').grid(row=11, column=val, columnspan=1, sticky=S + W)

        # set labels for algorithms
        self.l1 = Label(Frame1, text='Tracking algorithms', width=30, height=1, font=FontSize,
                        bg='light sea green')
        self.l1.grid(row=13, column=0, columnspan=3, sticky=S + W + E + N)

        # create and set the radio buttons for the choice of segmentation

        MethodV = IntVar()
        MethodV.set(0)  # initializing the choice, i.e. Centroid
        Methods = [('centroid', 0), ('shape', 1), ('shape + centroid', 2), ('opticalflow', 3)]

        def ShowCHoice():
            return MethodV.get()

        for txt, val in Methods:
            if val <= 2:
                Radiobutton(Frame1,
                            text=txt,
                            padx=20,
                            variable=MethodV,
                            command=ShowCHoice,
                            value=val, width=13, height=1, font=FontSize2, bg='white',
                            activeforeground='blue').grid(row=14, column=val, columnspan=1, sticky=S + W)
            if val > 2:
                Radiobutton(Frame1,
                            text=txt,
                            padx=20,
                            variable=MethodV,
                            command=ShowCHoice,
                            value=val, width=13, height=1, font=FontSize2, bg='white',
                            activeforeground='blue').grid(row=15, column=0, columnspan=1, sticky=S + W)

        self.SubmitButton = Button(Frame1, width=16, height=1, font=FontSize2, bg='white', text='Apply',
                                   command=self.process_filename, state="normal")
        self.SubmitButton.grid(row=16, column=1, columnspan=1, sticky=S + W + E + N)

        self.pbar_f = ttk.Progressbar(Frame1, orient="horizontal", length=600, mode="determinate")
        self.pbar_f.grid(row=17, column=0, columnspan=3, sticky=S + W + E + N)

        self.l1 = Label(Frame1, text='Other functions', width=30, height=1, font=FontSize,
                        bg='light sea green')

        self.l1.grid(row=18, column=0, columnspan=3, sticky=S + W + E + N)
        self.Btn1 = Button(Frame1, text='show trajectories', command=self.get_gif, width=16, height=1,
                           font=FontSize2, bg='white')
        self.Btn1.grid(row=19, column=0, columnspan=1, sticky=S + W)

        self.Btn1 = Button(Frame1, text='generate a GIF', command=self.generate_gif, width=18, height=1,
                           font=FontSize2,
                           bg='white')
        self.Btn1.grid(row=19, column=1, columnspan=1, sticky=S + W)
        self.Btn1 = Button(Frame1, text='plot # cell per frame', command=self.get_gif, width=17, height=1,
                           font=FontSize2,
                           bg='white')
        self.Btn1.grid(row=19, column=2, columnspan=1, sticky=S + W)
        # self.Btn1 = Button(Frame1, text='preview A sample', command=self.get_gif, width=16, height=1, font=FontSize2,
        #                   bg='white')
        # self.Btn1.grid(row=18, column=1, columnspan=1, sticky=S + W)
        self.Btn1 = Button(Frame1, text=' download cell locations', command=self.download_file, width=18, height=1,
                           font=FontSize2,
                           bg='white')

        self.Btn1.grid(row=20, column=1, columnspan=1, sticky=S + W)
        self.Btn1 = Button(Frame1, text='download trajectories', command=self.file_save, width=16, height=1,
                           font=FontSize2,
                           bg='white')
        self.Btn1.grid(row=20, column=0, columnspan=1, sticky=S + W)

    def selected(self):
        if self.CheckVar1.get() == 1:
            print "do something 1"
        elif self.CheckVar1.get() == 2:
            print "do something 2 "
        elif self.CheckVar1.get() == 3:
            print "do something 3 "
        elif self.CheckVar1.get() == 4:
            print "do something 4 "
        elif self.CheckVar1.get() == 5:
            print "do something 5 "
        elif self.CheckVar1.get() == 6:
            print "do something 6 "
        else:
            print "do nothing"

    def process_input(self):
        print 'sami'

    # load files
    def load_file(self):
        fname = askopenfilename(filetypes=(("VIDEO files", "*.avi"),
                                           ("GIF files", "*.gif"),
                                           ("TIF files", "*.tif"),
                                           ("PNG  files", "*.png"),
                                           ("JPEJ files", "*.jpg"),
                                           ("ALL files", "*.*")))
        if fname:
            try:
                self.filename = fname
                tkMessageBox.showinfo('File name', 'you selected {0}'.format(self.filename))

                # read avi format
                if '.avi' in self.filename:
                    self.frames, _ = read_avi(self, self.filename)

                # read tiff format
                if '.tif' in self.filename:
                    tkMessageBox.showinfo('reading...', 'processing frames')
                    tkMessageBox.ABORT
                    self.frames = read_tiff(self, self.filename)
                    tkMessageBox.showinfo('Completed', 'reading frames completed')


            except:  # <- naked except is a bad idea
                tkMessageBox.showinfo('Open Source File', 'Failed to read file{0}'.format(fname))
        else:
            self.filename = []
            tkMessageBox.showinfo('File name', 'no file is selected')
            # return fname

    def save_file(self):
        myFormats = [
            ('Windows Bitmap', '*.bmp'),
            ('Portable Network Graphics', '*.png'),
            ('JPEG / JFIF', '*.jpg'),
            ('CompuServer GIF', '*.gif'),
        ]

        fileName = asksaveasfilename(initialdir=overlaytrajectoryanidir, filetypes=myFormats,
                                     title="Save the image as...")
        if len(fileName) > 0:
            print "Now saving under %s" % nomFichier

        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        text2save = str(text.get(1.0, END))  # starts from `1.0`, not `0.0`
        f.write(text2save)
        f.close()  # `()` was missing

    def download_file(self):
        dirname = tkFileDialog.askdirectory(initialdir=csvdir, title='Please select a directory')
        if len(dirname) > 0:
            print "You chose %s" % dirname


    def callback(self):
        value1, value2, value3 = v.get(), v1.get(), v2.get()
        return value1, value2, value3

    def process_filename(self):

        segChoice = self.var.get()

        if segChoice == 0:
            tkMessageBox.showinfo('segmentation method', )
            segMethod = 'whiteBG'
        if segChoice == 1:
            segMethod = 'blackBK'
        else:
            segMethod = 'Lucas and Kanade'

        EstimatedCells, CellQuality, MinDistance = self.callback()

        shi_features = shi_segm(EstimatedCells, CellQuality, MinDistance)

        if self.frames:
            optical_flow(self, self.frames, self.frames[0], shi_features, segMeth=segMethod)

        else:
            tkMessageBox.showinfo('', 'no data to process, check your file if corrupted')
        # if MethodV.get() == 3:

        if '.png .jpeg' in self.filename:
            frames = read_others(self, self.filename)

    # display a gif file
    def get_gif(self):
        self.label1 = Label(bg="light cyan", width=950, height=300)
        self.label1.grid(row=0, column=2, columnspan=1, sticky=W + E + N + S)
        self.gifBackgroundImages = list()
        self.atualGifBackgroundImage = 0
        self.background = PhotoImage()
        animate(self, trackingdir)

    # get a trajector file
    def get_trajectory(self):

        im = Image(trajectorydic + 'finalTrajectory.png')

        photo = ImageTk.PhotoImage(file=trajectorydic + 'finalTrajectory.gif')
        # photo = Image("photo", file=trajectorydic+'finalTrajectory.png')
        self.label2 = Label(image=photo, width=300, height=400)
        self.label2.image = photo  # keep a reference!
        self.label2.grid(row=1, column=0, columnspan=1, sticky=W + E + N + S)

    def generate_gif(self):
        save_gif = True
        title = ''
        images, imgs = [], []
        for foldername in os.listdir(overlaytrajectorydir):
            images.append(foldername)
        images.sort(key=lambda x: int(x.split('.')[0]))

        for _, file in enumerate(images):
            print file
            im = img2.open(overlaytrajectorydir + dash + file)
            imgs.append(im)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

        im_ani = animation.ArtistAnimation(fig, ims, interval=600, repeat_delay=500, blit=False)

        if save_gif:
            im_ani.save(overlaytrajectoryanidir + 'animation.gif', writer='imagemagick')

    def file_open(self):
        """open a file to read"""
        # optional initial directory (default is current directory)
        initial_dir = csvdir
        # the filetype mask (default is all files)
        mask = \
            [("Text and Excel files", "*.txt *.csv "),
             ("HTML files", "*.htm"),
             ("All files", "*.*")]
        fin = askopenfile(initialdir=initial_dir, filetypes=mask, mode='r')
        text = fin.read()
        if text != None:
            self.text.delete(0.0, END)
            self.text.insert(END, text)

    def file_save(self):
        """get a filename and save the text in the editor widget"""
        # default extension is optional, here will add .txt if missing
        fout = asksaveasfile(mode='w', defaultextension=".csv")
        text2save = str(self.text.get(0.0, END))
        fout.write(text2save)
        fout.close()


if __name__ == "__main__":
    root = Tk()
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

    root.geometry('1500x1000')

    img = Image("photo", file="/home/sami/Desktop/multimot-logo-e1437119906276.png")
    root.tk.call('wm', 'iconphoto', root._w, img)
    app = MyFrame(master=root)
    app.mainloop()
