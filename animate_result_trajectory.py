import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import glob
import csv
import numpy as np
import cv2
from matplotlib import colors
import six
import colorsys
from pandas import DataFrame
import openpyxl

colors_ = list(six.iteritems(colors.cnames))
hex_ = [color[1] for color in colors_]
# Get the rgb equivalent.
rgb = [colors.hex2color(color) for color in hex_]
hsv = [colors.rgb_to_hsv(color) for color in rgb]

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def readlines():
    xlst = glob.glob('/home/sami/Desktop/code/segmentation/watershed_1.0_179.csv')

    with open(xlst[-1], 'rb') as f:
        reader = csv.reader(f)
        x_axis, y_axis, t_axis, CID, minValue = [], [], [], [], []

        for r, row in enumerate(reader):
            if r == 0:
                continue
            x = row[1]
            x = x[1:-1]
            x = x.split(',')
            x = map(float, x)
            ValueMin = len(x)
            minValue.append(ValueMin)

            y = row[2]

            y = y[1:-1]
            y = y.split(',')
            y = map(float, y)

            x_axis.append(x)
            y_axis.append(y)
            del x, y

    return x_axis, y_axis, minValue


filename = '/home/sami/9K5F98PS_F00000011.avi     '
cap = cv2.VideoCapture(filename)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
'''
width = int(cap.get(3))
fig, ax = plt.subplots(1, 1)
plt.ion()
plt.show()'''


def animate(xs, ys, minX):
    sizeX = len(xs)
    #for i in range(sizeX):
       # im = cv2.imread("/home/sami/Desktop/code/segmentation/images/Image_%d.png" % i)

    x2, y2, numCells, frameID = [], [], [], []
    minMatrix = min(minX)
    for i in range(sizeX):
        fig.clf()
        flag, frame = cap.read()
        x = xs[i]
        y = ys[i]
        #x = x[0:int(minMatrix)]
        #y = y[0:int(minMatrix)]
        '''
        x2.append(x)
        y2.append(y)

        x2 = x2[0]
        y2 = y2[0]

        x1 = np.array(x2)
        y1 = np.array(y2)
        old_x = x1.ravel()
        old_y = y1.ravel()'''



        N = len(x)
        count = 0
        HSV_tuples = [(B * 2.0 / N, 0.6, 0.6) for B in range(N)]
        RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray_image.shape, dtype="uint8")
        mask[:]=255
        for m,_ in enumerate(x):

            xx = x[m]
            yy = y[m]
            x2.append(int(xx))
            y2.append(int(yy))

            color2 = RGB_tuples[m]
            color3 = tuple([256*t for t in color2])
            font = cv2.FONT_HERSHEY_SIMPLEX

            #cv2.circle(mask, (int(xx), int(yy)),4, color3, 1)
            #cv2.putText(frame, "(%d, %d)" % (int(xx), int(yy)), (int(xx), int(yy)), font, 1, color3, 2, cv2.LINE_AA)
            #cv2.imshow("frame %d" % int(i), mask)
            #cv2.waitKey(33)
            #cv2.destroyAllWindows()

        if i == 0 :
            oldXPoints = x2
            oldYPoints = y2
            continue

        N = len(x2)
        HSV_tuples = [(B * 100.0 / N, 100.0, 20.0) for B in range(N)]
        RGB_tuples = map(lambda B: colorsys.hsv_to_rgb(*B), HSV_tuples)

        for ii,_ in enumerate(x2):
            newX = x2[ii]
            newY = y2[ii]
            color2 = RGB_tuples[ii]
            color3 = tuple([256 * t for t in color2])
            cv2.line(mask, (int(newX), int(newY)), (int(newX), int(newY)), color3, 3)
        cv2.imshow('mask %d' % int(i), mask)
        name = "/home/sami/cellMove/cellMove/F %d.jpg" % i
        cv2.imwrite(name, mask)
        print count
        count += 1
        cv2.waitKey(0)
        oldXPoints = x2
        oldYPoints = y2
        del x2, y2
        x2, y2 = [], []

        cv2.destroyAllWindows()


#        cv2.line(frame, (int(xx), int(yy)), (int(old_x), int(old_y)), color,5,2)
                #cv2.putText(img, 'CellID %d', (10, 500), font, 1, (255, 255, 255), 2)
        #cv2.imshow("frame %d" % int(i), frame)

        #cv2.waitKey(33)
        #time.sleep(10)
        #cv2.imwrite("/home/sami/Desktop/code/segmentation/images/ImageCentroid_%d.png" % i, im)

        numCells.append(N)
        frameID.append(i)
    #df = DataFrame({'Frame Number': frameID, 'Number of Cells': numCells})
    #df.to_excel('/home/sami/Desktop/code/segmentation/CANCERCELL_NBG.xlsx', sheet_name='sheet1', index=False)


xss, yss, minXX = readlines()
animate(xss,yss, minXX)
'''
def animate(i):

    xr, yr = readlines()

    x = xr[i]
    print x

    y = yr[i]
    print y
    exit()
    ax1.clear()
    ax1.plot(x, y)
    time.sleep(3)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()'''