import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from scipy.misc import imread
import matplotlib.cbook as cbook
import colorsys


df = pd.read_csv('/home/sami/Downloads/CellMigration-master/Trajectories_Duisburg-Essen.csv')
saved_column = df['CellID'] #you can also use df['column_name']
saved_column2 = df['x-axis'] #you can also use df['column_name']
saved_column3 = df['y-axis'] #you can also use df['column_name']

f = open('/home/sami/Desktop/code/segmentation/watershedBGSUB_iMMUNE.0_179.csv', 'rb')
reader = csv.reader(f)
xs, ys, ids = [],[],[]
for i, row in enumerate(reader):

    if i > 0:
        ids.append(row[0])
        r1 = row[1].replace('[','')
        r2 = r1.replace(']', '')
        splitedX = r2.split(',')
        xx = map(float, splitedX)
        # manipulate the data to convert to the right data type
        r3 = row[2].replace('[','')
        r4 = r3.replace(']', '')
        splitedY = r4.split(',')
        yy = map(float, splitedY)
        # append to the data
        xs.append(xx)
        ys.append(yy)
f.close()


datafile = '/home/sami/Desktop/code/segmentation/images/Image_179.png'
img = imread(datafile)
for tt, ii in enumerate(xs):

    x = xs[tt]
    y = ys[tt]
    plt.scatter(x,y)

plt.imshow(img)
plt.show()


path = ''
def load_trajectory(path):
    return path
