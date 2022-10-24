import os
import sys
import numpy
import pandas
import matplotlib.pyplot as plt
from test import DPC

path = "../dataSet/"
PATH = "../dataSet/"
DATAS = ['spiral.dat']


def show_data():
    """
    """
    files = os.listdir(path)
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97,
                        bottom=0.03, hspace=0.2, wspace=0.2)
    i = 0
    j = 0
    for file in files:
        data_path = path+file
        data_title = os.path.splitext(file)[0]
        points = pandas.read_csv(data_path, sep="\t", usecols=[0, 1])
        axes[i][j].scatter(points.loc[:, 'x'], points.loc[:, 'y'], s=1)
        axes[i][j].set_title(data_title)
        if j == 4:
            i += 1
            j = 0
        else:
            j += 1
    plt.show()


if __name__ == "__main__":
    """"""
    # show_data()
    dpc = DPC(PATH+DATAS[0], DATAS[0], num=3, dc_percent=4)
    dpc.cluster()
