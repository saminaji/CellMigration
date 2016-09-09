import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import sys
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import measure
import time
import sys, os
sys.path.append('/home/sami/Desktop/code/segmentation/snakeSegmentation.py')




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

#path = '/home/sami/9K5F98PS_F00000011.avi'
path = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
  # params for ShiTomasi corner detection

feature_params = dict(maxCorners=300,
                      qualityLevel = 0.5,
                      minDistance = 20,
                      blockSize = 7 )

'''feature_params = dict(maxCorners=100,
                      qualityLevel = 0.5,
                      minDistance = 50,
                      blockSize = 7 )'''

  # Parameters for lucas kanade optical flow

lk_params = dict(winSize=(10, 10), maxLevel = 2,

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  # Create some random colors

color = np.random.randint(0, 255, (200, 3))

# Take first frame and find corners in it
frames, _ = filereader(path)
old_frame = frames[0]



old_gray_image = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('sss', old_gray_image)
old_gray_image = cv2.equalizeHist(old_gray_image)
cv2.imshow('ddd',old_gray_image)
cv2.waitKey(0)


p0 = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

  # Create a mask image for drawing purposes

mask = np.ones_like(old_frame)

length2 = len(frames)
for ii, frame in enumerate(frames):
    try:

        new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_gray_image = cv2.equalizeHist(new_gray_image)

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

            #mask = cv2.line(mask, (a, b), (c, d), (255,255,255), 2)

            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            #frame = cv2.circle(frame, (a, b), 5, (255,255,255), -1)

            frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_PLAIN, 1,color[i].tolist())
            #frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255))


            img = cv2.add(frame, mask)
            edged2 = np.hstack([mask, img])

            if initialValue < cont-1:
                cv2.imshow('frame', edged2)
                cv2.waitKey(33)
                time.sleep(0.05)

            if initialValue == cont:
                cv2.imshow('frame', edged2)
                cv2.waitKey(0)

            initialValue +=1

              # Now update the previous frame and previous points
            old_gray_image = new_gray_image.copy()
            p0 = good_new.reshape(-1, 1, 2)
        if ii == length2-1:
            cv2.imshow('frame', edged2)
            cv2.waitKey(0)

        cv2.imwrite('/home/sami/Desktop/optflow/tr_%d.png'%ii,img)
    except EOFError:
        continue
cv2.destroyAllWindows()
