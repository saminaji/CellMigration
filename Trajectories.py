import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as animation
import numpy as np
import time
import glob
import csv
from pylab import *


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(40,50), ylim=(5, 15))
line, = ax.plot([], [])


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# grab the data from the repository
def getdata():
    xlst = glob.glob('/home/sami/Desktop/code/segmentation/data_shape6.1_0.csv')
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
def animate(i):
    x, y = [], []
    x_i, y_i, _, ids = getdata()
    indices = [int(t) for t, j in enumerate(ids) if int(j) == i]

    for index in indices:
        x.append(x_i[index])
        y.append(y_i[index])

    line.set_data(x[:-1], y[:-1])
    time.sleep(20)

    return line,

_, _, _, ids = getdata()
frameNum = len(np.unique(ids))
print frameNum

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=int(frameNum), interval=800)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
