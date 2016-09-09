import matplotlib.pyplot as plt
import csv
import glob
import cv2


def plotTrajector():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    plt.ion()
    plt.show(block=True)

    verts = [
        (0, 0),
        # I'm just assuming two sets of points here. I actually intend to put variables here which I can update in real time.
        (27, 0)
    ]

    codes = [Path.MOVETO,
             Path.LINETO]

    path = Path(verts, codes)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='white', lw=2)
    ax.add_patch(patch)
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-100,100)
    plt.draw()
    time.sleep(1)

def plotMovements(xs, ys, ts):
    plt.plot(xs, ys, color='k', linestyle='-', linewidth=1)
#    a = xs.get_xticks().tolist()
    # axes.set_xticklabels(a)
    plt.xticks(ts)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prepare():

    xlst = glob.glob('/home/sami/Desktop/code/segmentation/*.csv')

    with open(xlst[-1], 'rb') as f:
        reader = csv.reader(f)
        x_axis, y_axis, t_axis = [], [], []

        for row in reader:
            print row
            x_axis.append(row[2])
            y_axis.append(row[3])
            t_axis.append(row[4])

    return x_axis, y_axis,t_axis

def readframes(path):
    cap = cv2.VideoCapture(filesdirectory)
    timestamp = []
    frames = []
    try:
        while cap.isOpened():
            _, img = cap.read()
            # get the frame in seconds
            t1 = cap.get(0)
            timestamp.append(t1)
            if img is None:
                break
            frames.append(img)
    except EOFError:
        pass
    return timestamp, frames


# main function

path = '/home/sami/9K5F98PS_F00000011.avi'

t, f = readframes(path)

x, y, t = prepare()


x = x[1:120]
print x
exit()
y = y[1:120]
print y
t = t[1:120]
print t

plotMovements(x, y, t)
