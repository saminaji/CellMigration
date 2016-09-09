
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import csv
import numpy as np
import re
import string
def readfile():
    with open('watershed_1.0_179.csv', 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()


        column = {h:[] for h in headers}
        X,Y = [], []
        for row in reader:
            x = row[1]
            X.append(x)
            y = row[2]
            Y.append(y)
            for h, v in zip(headers, row):
                column[h].append(v)
    return X, Y


def prepareData (par1, par2):

    lenX = len(par1)
    Xs, Ys = [], []
    for i in range(lenX):

        tmp1 = par1[i]
        print tmp1

        tmp2 = par2[i]

        #tmp1 = int(tmp1)
        #tmp2 = int(tmp2)

       # Xs.append(tmp1)
       # Ys.append(tmp2)
    return Xs, Ys


xs, ys = readfile()
numFrames = len(xs)

# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, numFrames)])

for i in range(numFrames):

    mylst = map(lambda each: xs[0].strip('[]'), xs[i])
    myList =  string.split(mylst[0],',')
    mylst2 = map(lambda each: ys[0].strip('[]'), ys[i])
    myList2 = string.split(mylst2[0], ',')

    #xPlot, yPLot = prepareData(xs[i], ys[i])
    plt.plot(myList, myList2, '--')
    plt.show()
    plt.hold(True)
    #pause(2)



