

#load  packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import seaborn as sns
import glob
import random

COLORS = [(139, 0, 0),
          (0, 100, 0),
          (0, 0, 139)]

rgbint = 129
def random_color():
    return random.choice(COLORS)

def VideoFrameReaders(VideoDirectory):
    cap = cv2.VideoCapture(VideoDirectory)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    timestamp = []
    count = 0
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            time = cap.get(0) #get the frame in seconds
            timestamp.append(time)

            print timestamp

            if frame == None:
                break;
            image = frame
            segments = slic(img_as_float(image), n_segments=100, sigma=5)

            # # show the output of SLIC
            # fig = plt.figure("Superpixels")
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
            # plt.axis("off")
            # plt.show()
            print("[INFO] {} unique segments found".format(len(np.unique(segments)) - 1))

            for (i, segVal) in enumerate(np.unique(segments)):
                # construct a mask for the segment
                print "[x] inspecting segment %d" % (i)
                mask = np.zeros(image.shape[:2], dtype="uint8")
                mask[segments == segVal] = 255

                # show the masked region
                cv2.imshow("Mask", mask)
                cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
                cv2.waitKey(0)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except EOFError:
        pass
    return count,timestamp,

path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');

c,t = VideoFrameReaders(path[0])



# get frames from a directory
#frames = glob.glob('/home/sami/Desktop/movies/extractFrames/*.jpg')
