import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from sklearn import neighbors
from scipy.sparse import csgraph
from scipy import ndimage as ndi
import time
from skimage.feature import canny
from libtiff import TIFF

try:
    from cv2 import cv
except Exception:
    print "cant import cv"

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

def aut_canny(image):
#    seg = canny(image/255.)
    seg = cv2.Canny(image, 100,193)
    return seg





path = '/home/sami/Downloads/ctr_multipage (2).tif'
#path = '/home/sami/9K5F98PS_F00000011.avi'

frames = readtiff(path)
#frames,_ =filereader(path)

old_frame = frames[0]

  # params for ShiTomasi corner detecto

feature_params = dict(maxCorners=120,
                      qualityLevel = 0.5,
                      minDistance = 40,
                      blockSize = 7 )

  # Parameters for lucas kanade optical flow

lk_params = dict(winSize=(20, 20), maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))


  # Create some random colors

color = np.random.randint(0, 255, (200, 3))

old_gray_image1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray_image = cv2.equalizeHist(old_gray_image1)

def shi_tomasi(image, feature_params, img2):

    # detect corners in the image
    #old_gray_image = histogram_equaliz(image)
    corners = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
    #corners = np.float32(corners)

    # go through the corner
    for corner in corners:

        x, y = corner[0]
        print x,  y
        cv2.circle(img2, (x,y),5, (0,0,255),-1)
        cv2.imshow('Best Corner', img2)
        cv2.waitKey(33)
    return corners

def harris_corner (blurred, image):
    corners = cv2.goodFeaturesToTrack(blurred,  # img
                            150,  # maxCorners
                            0.09,  # qualityLevel
                            30,  # minDistance
                            None,  # corners,
                            None,  # mask,
                            7,  # blockSize,
                            useHarrisDetector=True,  # useHarrisDetector,
                            k=0.05  # k
                            )
    for corner in corners:

        x, y = corner[0]
        print x,  y
        cv2.circle(image, (x,y),5, (0,0,255),-1)
        cv2.imshow('Best Corner', image )
        cv2.waitKey(33)

    return corners

#initialPoints = harris_corner(old_gray_image,old_gray_image1)
#exit()
initialPoints = shi_tomasi(old_gray_image,feature_params, old_gray_image1)
#exit()
#initialPoints = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)

# Create a mask image for drawing purposes
height,width,depth = old_frame.shape
mask = np.zeros_like(old_frame,)



for ii, frame in enumerate(frames):
    try:

        new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_gray_image = cv2.equalizeHist(new_gray_image)


        # calculate optical flow

        newPoints, st, err = cv2.calcOpticalFlowPyrLK(old_gray_image, new_gray_image, initialPoints, None, **lk_params)

       # newPoints, st, err = cv.CalcOpticalFlowHS(old_gray_image, new_gray_image, False, uv[0], uv[1],
       #                      smoothing,
        #                     (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 8, 0.1))
          # Select good points

        good_new = newPoints[st == 1]

        good_old = initialPoints[st == 1]

          # draw the tracks
        cont = len(good_new)
        initialValue = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):

            a, b = new.ravel()
            print a, b

            c, d = old.ravel()
            print c, d

            #mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            mask = cv2.line(mask, (a, b), (c, d), (255,255,255), 2)

#            maskPlot = cv2.line(maskPlot, (a, b), (c, d), color[i].tolist(), 2)
            #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            #frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,color[i].tolist())
            frame = cv2.putText(frame,"%d"%i,(a,b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255))

            img = cv2.add(frame, mask)


            edged2 = np.hstack([mask, img])

            if initialValue < cont - 1:
                cv2.imshow('frame', edged2)
#                cv2.imshow('plot', maskPlot)
                cv2.waitKey(33)
            if initialValue == cont:
                cv2.imshow('frame', edged2)
                cv2.waitKey(0)

            initialValue += 1

              # Now update the previous frame and previous points
            old_gray_image = new_gray_image.copy()

            initialPoints = good_new.reshape(-1, 1, 2)
        cv2.imwrite('/home/sami/Desktop/code/segmentation/08_08_2016_12_31/tracking/%d.png'%ii, edged2)
    except EOFError:
        continue
cv2.destroyAllWindows()
