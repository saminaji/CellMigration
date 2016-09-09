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

colors_ = list(six.iteritems(colors.cnames))
hex_ = [color[1] for color in colors_]
# Get the rgb equivalent.
rgb = [colors.hex2color(color) for color in hex_]
hsv = [colors.rgb_to_hsv(color) for color in rgb]

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def readlines():
    xlst = glob.glob('/home/sami/Desktop/code/segmentation/watershed_CANCER2.0_71.csv')

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


#filename = '/home/sami/9K5F98PS_F00000011.avi'
filename = '/home/sami/Desktop/movies/movies/9Z56H8T9_F00000003.avi'
cap = cv2.VideoCapture(filename)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
'''
width = int(cap.get(3))
fig, ax = plt.subplots(1, 1)
plt.ion()
plt.show()'''


def animate(xs, ys, minX):
    sizeX = len(xs)
    print sizeX

    #for i in range(sizeX):
       # im = cv2.imread("/home/sami/Desktop/code/segmentation/images/Image_%d.png" % i)

    x2, y2 = [], []
    minMatrix = min(minX)
    for i in range(frames):
        fig.clf()
        flag, frame = cap.read()
        x = xs[i]
        y = ys[i]
        x = x[0:int(minMatrix)]
        y = y[0:int(minMatrix)]
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

            #old_x = xx
           # old_y = yy


            #print len(old_y)
            #if i == 0:
            color2 = RGB_tuples[m]
            color3 = tuple([256*t for t in color2])
            font = cv2.FONT_HERSHEY_SIMPLEX

#            cv2.line(mask, (int(xx), int(yy)), (int(old_x), int(old_y)), color, 5, 2)
            if i >0:
                cv2.putText(frame, "(%d, %d)" % (int(xx), int(yy)), (int(xx), int(yy)), font, 1, color3, 2, cv2.LINE_AA)
                cv2.imshow("frame %d" % int(i), frame)
                name = "/home/sami/framesPPT/nextFrame/CancerBG_Cell_ %d.jpg" % m
                cv2.imwrite(name, frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        startPoints =  x2

#        cv2.line(mask, (int(xx), int(yy)), (x2, y2), color, 5, 2)
       # name = "/home/sami/cellMove/Move%d.jpg" % count
       # cv2.imwrite(name, mask)
        #count += 1
       # cv2.waitKey(0)

      #  cv2.destroyAllWindows()
       # print old_x


       # cv2.line(frame, (int(xx), int(yy)), (int(old_x), int(old_y)), color,5,2)
                #cv2.putText(img, 'CellID %d', (10, 500), font, 1, (255, 255, 255), 2)
       # cv2.imshow("frame %d" % int(i), frame)

        #cv2.waitKey(33)
        #time.sleep(10)
        #cv2.imwrite("/home/sami/Desktop/code/segmentation/images/ImageCentroid_%d.png" % i, im)


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