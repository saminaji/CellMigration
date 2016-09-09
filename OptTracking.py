import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import time
from libtiff import TIFF

def readtiff(path):
    frames = []
    tif = TIFF.open(path, mode='r')

    try:
        for cc, tframe in enumerate(tif.iter_images()):
            frames.append(tframe)
    except EOFError:
        pass
    return frames

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
    return frames, timestamp


def update(dummy=None):
    if seed_pt is None:
        cv2.imshow('floodfill', img)
        return
    flooded = img.copy()
    mask[:] = 0
    lo = cv2.getTrackbarPos('lo', 'floodfill')
    hi = cv2.getTrackbarPos('hi', 'floodfill')
    flags = connectivity
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,) * 3, (hi,) * 3, flags)
    cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
    cv2.imshow('floodfill', flooded)

def watershed_segmentation(image):

    # pre-process the image before performing watershed
    # load the image and perform pyramid mean shift filtering to aid the thresholding step
    # shifted = cv2.pyrMeanShiftFiltering(closing, 15, 39)
    im = cv2.threshold(image, 173, 255, cv2.THRESH_BINARY)
    im = im[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(im, kernel, iterations=2)
    Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(Gradient, cv2.MORPH_CLOSE, kernel)

    shifted = cv2.pyrMeanShiftFiltering(closing, 10, 20)
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

        # otherwise, allocate memory for the label region and draw
        # it on the mask

        # mask = np.zeros(gray.shape, dtype="uint8")
        masks2[labels == label] = 255



        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #masks2 = cv2.floodFill(image.copy(), masks2, (3, 0), 0, 0, 0, flags = 4 | cv2.FLOODFILL_MASK_ONLY)
        masks2 = cv2.morphologyEx(masks2, cv2.MORPH_CLOSE, kernel)



        # dilation = cv2.dilate(mask, kernel, iterations=2)
        # Gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_OPEN, kernel)
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_CLOSE, kernel)
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_DILATE, kernel)
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_OPEN, kernel)
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_CLOSE, kernel)

    return masks2

def aut_canny(image):

    seg = cv2.Canny(image, 100,193)
    return seg

def auto_histogram(image):

    return image

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

path = '/home/sami/Downloads/ctr_multipage (2).tif'
#path = '/home/sami/9K5F98PS_F00000011.avi'

frames = readtiff(path)
#frames,_ =filereader(path)

old_frame = frames[0]



#cap = cv2.VideoCapture('/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi')
#cap = cv2.VideoCapture('/home/sami/9K5F98PS_F00000011.avi')
#cap = cv2.VideoCapture('/home/sami/Downloads/ctr_multipage (2).tif')

  # params for ShiTomasi corner detection

feature_params = dict(maxCorners=50,
                      qualityLevel = 0.1,
                      minDistance = 5,
                      blockSize = 7 )

  # Parameters for lucas kanade optical flow

lk_params = dict(winSize=(20, 20), maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

  # Create some random colors

color = np.random.randint(0, 255, (200, 3))

# Take first frame and find corners in it

#ret, old_frame = cap.read()
old_gray_image = old_frame
'''
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

old_gray_image = cv2.threshold(old_gray_image, 173, 255, cv2.THRESH_BINARY)
old_gray_image = old_gray_image[1]

old_gray_image = cv2.morphologyEx(old_gray_image, cv2.MORPH_CLOSE, kernel)
old_gray_image = cv2.morphologyEx(old_gray_image,cv2.MORPH_DILATE,kernel)
old_gray_image = aut_canny(old_gray_image)

cv2.imshow("ss", old_gray_image)
cv2.waitKey(0)
'''
old_gray_image = watershed_segmentation(old_frame)
cv2.imshow("ss", old_gray_image)
cv2.waitKey(0)
#old_gray_image = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)



p0 = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

  # Create a mask image for drawing purposes

mask = np.zeros_like(old_frame)

for i, frame in enumerate(frames):
    try:

        '''new_gray_image = cv2.threshold(frame, 173, 255, cv2.THRESH_BINARY)
        new_gray_image = new_gray_image[1]

        new_gray_image = cv2.morphologyEx(new_gray_image, cv2.MORPH_CLOSE, kernel)
        new_gray_image = aut_canny(new_gray_image)'''

        new_gray_image  = watershed_segmentation(frame)
       # new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          # calculate optical flow

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_image, new_gray_image, p0, None, **lk_params)

          # Select good points

        good_new = p1[st == 1]

        good_old = p0[st == 1]

          # draw the tracks
        cont = len(good_new)
        initialValue = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):

            a, b = new.ravel()
            print a, b

            c, d = old.ravel()
            print c, d

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,color[i].tolist())

            img = cv2.add(frame, mask)


            if initialValue < cont:
                cv2.imshow('frame', img)

                k = cv2.waitKey(33) & 0xff
            if initialValue == cont:
                cv2.imshow('frame', img)
                k = cv2.waitKey(0) & 0xff

            initialValue +=1
           # if k == 27:

            #    break

              # Now update the previous frame and previous points
            old_gray_image = new_gray_image.copy()
            #p0 = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

            p0 = good_new.reshape(-1, 1, 2)

    except EOFError:
        continue
cv2.destroyAllWindows()
#cap.release()