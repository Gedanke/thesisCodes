# -*- coding: utf-8 -*-

import os
import sys
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import *
from sklearn.metrics import *
from dpc import *
from test import *

PATH = "../../dataSet/"
DATAS = ['test.dat']


def deal_data():
    """
    数据调整
    Returns
    -------

    """
    files = os.listdir(PATH + "raw/")
    for file in files:
        '''文件路径'''
        data_path = PATH + "raw/" + file
        data_title = os.path.splitext(file)[0]
        '''读取数据'''
        data = pandas.read_csv(data_path, sep="\t")
        '''获取列名'''
        columns_name = list(data.columns)
        '''将第一列列名修改为 x，第一列列名修改为 y，最后一列列名修改为 num'''
        data = data.rename(columns={columns_name[0]: "x", columns_name[1]: "y", columns_name[-1]: "num"})
        '''保存结果'''
        data.to_csv(PATH + "data/" + data_title + ".csv", index=False)


def show_data_():
    """
    """
    files = os.listdir(PATH)
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97,
                        bottom=0.03, hspace=0.2, wspace=0.2)
    i = 0
    j = 0
    for file in files:
        data_path = PATH + file
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
    deal_data()
