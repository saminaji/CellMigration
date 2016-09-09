import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as animation
import numpy as np
import time
import glob
import csv
from pylab import *
import cv2


# First set up the figure, the axis, and the plot element we want to animate
'''fig = plt.figure()
ax = plt.axes(xlim=(40, 50), ylim=(5, 15))
line, = ax.plot([], [])'''


# grab the data from the repository
def getdata():
    xlst = glob.glob('/home/sami/Desktop/code/segmentation/data_shape5_0.csv')
    print xlst

    with open(xlst[-1], 'rb') as f:
        reader = csv.reader(f)
        x_axis, y_axis, t_axis, CID = [], [], [], []

        for row in reader:
            CID.append(row[1])
            x_axis.append(row[2])
            y_axis.append(row[3])
            t_axis.append(row[4])
    return x_axis[1:], y_axis[1:], t_axis[1:], CID[1:]


# animation function.  This is called sequentially
def animate(ids, xs, ys, ts):
    sizeX = len(xs)
    for i in range(sizeX):
        if i is 0:
            continue
        im = cv2.imread("/home/sami/Desktop/code/segmentation/images/Image_%d.png" % i)

        x = xs[i]
        y = ys[i]
        print ts[i]

        print x, y
        color = (0, 255, 255)

        #cv2.circle(im, (int(x), int(y)), 4, color)
        cv2.line(im, (int(xs[0]), int(ys[0])), (int(xs[-1]), int(ys[-1])), color,5)
        #cv2.putText(img, 'CellID %d', (10, 500), font, 1, (255, 255, 255), 2)
        cv2.imshow("frame %d" % int(i), im)

        cv2.waitKey(0)
        #time.sleep(10)
        #cv2.imwrite("/home/sami/Desktop/code/segmentation/images/ImageCentroid_%d.png" % i, im)


xs, ys, ts, ids = getdata()
print xs
print ys
animate(ids, xs, ys,ts)




# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
