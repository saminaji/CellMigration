import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from scipy.misc import imread
import matplotlib.cbook as cbook
import colorsys


f = open('/home/sami/Desktop/code/segmentation/Trajectories.csv','rb')
#f = open('/home/sami/Downloads/Trajecctories_UGerman_0.csv', 'rb')
reader = csv.reader(f)
xs, ys, ids = [],[],[]
for i, row in enumerate(reader):

    if i > 0:

        ids.append(row[1])
        xs.append(row[2])
        ys.append(row[3])

f.close()


id = np.unique(ids)

datafile = '/home/sami/Desktop/code/segmentation/Image_518.png'
datafile = '/home/sami/Desktop/code/segmentation/images/Image_179.png'

img = imread(datafile)
N = len(id)
HSV_tuples = [(B * 2.0 / N, 0.6, 0.6) for B in range(N)]
RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)

for tt, ii in enumerate(id):
    testid = [i for i,x in enumerate(ids) if x == ii]
    startIndex = min(testid)
    endIndex = max(testid)
    color2 = RGB_tuples[tt]
    color3 = tuple([256 * t for t in color2])

    x = xs[startIndex:endIndex]
    y = ys[startIndex:endIndex]
    plt.plot(x,y,color=color2)

plt.imshow(img)
plt.show()


path = ''
def load_trajectory(path):
    return path
