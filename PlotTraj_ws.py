import csv
import glob
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import datetime

# create a directory for each submission
now = datetime.datetime.now()
dirLast = str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)



def getdata(path):
    xlst = glob.glob(path)
    print xlst

    with open(xlst[-1], 'rb') as f:
        reader = csv.reader(f)
        x_axis, y_axis, t_axis, FID = [], [], [], []

        for i, row in enumerate(reader):
            if i > 0:
                newstr = row[1].replace('[', '')
                newstr = newstr.replace(']', '')
                newstr2 = map(float, newstr.split(','))
                newstr2 = map(int, newstr2)

                newstr3 = row[2].replace('[', '')
                newstr3 = newstr3.replace(']', '')
                newstr4 = map(float, newstr3.split(','))
                newstr4 = map(int, newstr4)

                FID.append(row[0])
                x_axis.append(newstr2)
                y_axis.append(newstr4)
                t_axis.append(row[3])
                print x_axis, y_axis
                plt.plot(x_axis, y_axis)
                plt.show()
                exit()

                for i,_ in enumerate(newstr2):

                    x = int (newstr2[i])
                    y = int (newstr4[i])
                    print x, y
                    plt.plot(x, y,'k-', lw=4)
                    plt.hold(True)
                    plt.show()

    return x_axis[1:], y_axis[1:], t_axis[1:], FID[1:]

path = '/home/sami/Desktop/code/segmentation/watershed_1.0_179.csv'
Xtrajectory, Ytrajectory,_,_ = getdata(path)


print Xtrajectory




