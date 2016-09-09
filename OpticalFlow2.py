import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import time
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import measure


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

def watershed_segmentation(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 15, 39)
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


    masks2 = np.zeros(gray.shape, dtype="uint8")
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it

        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        # mask = np.zeros(gray.shape, dtype="uint8")
        masks2[labels == label] = 255
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #masks2 = cv2.morphologyEx(masks2, cv2.MORPH_OPEN, kernel)
       # masks2 = cv2.morphologyEx(masks2, cv2.MORPH_CLOSE, kernel)

    return masks2


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # masks2 = cv2.morphologyEx(masks2, cv2.MORPH_OPEN, kernel)
    tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, kernel)
    #wide = cv2.morphologyEx(wide,cv2.MORPH_OPEN,kernel)
    wide = cv2.morphologyEx(wide,cv2.MORPH_CLOSE,kernel)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    edged2 = np.hstack([wide, tight, edged])

    # return the edged image
    return tight

def shapesegment(image):

    segments = slic(img_as_float(old_frame), n_segments=100, sigma=0.5)
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask[segments == segVal] = 255


    return mask

#cap = cv2.VideoCapture('/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi')
#cap = cv2.VideoCapture('/home/sami/9K5F98PS_F00000011.avi')

path = '/home/sami/9K5F98PS_F00000011.avi'
path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
  # params for ShiTomasi corner detection

feature_params = dict(maxCorners=50,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 9 )

  # Parameters for lucas kanade optical flow

lk_params = dict(winSize=(10, 10), maxLevel = 2,

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  # Create some random colors

color = np.random.randint(0, 255, (200, 3))

# Take first frame and find corners in it
frames, _ = filereader(path)
old_frame = frames[0]

#old_gray_image = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

old_gray_image = watershed_segmentation(old_frame)


p0 = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

  # Create a mask image for drawing purposes

mask = np.ones_like(old_frame)


for ii, frame in enumerate(frames):
    try:


        new_gray_image  = watershed_segmentation(frame)
        #new_gray_image = auto_canny(frame, sigma=0.33)
        new_gray_image =   cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

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
            frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_PLAIN, 1,color[i].tolist())

            img = cv2.add(frame, mask)
            edged2 = np.hstack([mask, img])

            if initialValue < cont-1:
                cv2.imshow('frame', edged2)

                cv2.waitKey(33)
            if initialValue == cont:
                cv2.imshow('frame', edged2)
                cv2.waitKey(0)

            initialValue +=1

            # Now update the previous frame and previous points
            old_gray_image = new_gray_image.copy()
            p0 = good_new.reshape(-1, 1, 2)
        #cv2.imwrite('/home/sami/Desktop/optflow/Cancer_%d.png'%ii,edged2)
    except EOFError:
        continue
cv2.destroyAllWindows()
#cap.release()