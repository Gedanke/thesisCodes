# -*- coding: utf-8 -*-

import os
import sys
import copy
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import *
from sklearn.metrics import *
from scipy.optimize import linear_sum_assignment as linear_assignment


def hungarian_cluster_acc(x, y):
    """
    https://blog.csdn.net/weixin_44839047/article/details/121885467
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    m = 1 + max(x.max(), y.max())
    n = len(x)
    total = numpy.zeros([m, m])
    for i in range(n):
        total[x[i], int(y[i])] += 1
    w = total.max() - total
    w = w - w.min(axis=1).reshape(-1, 1)
    w = w - w.min(axis=0).reshape(1, -1)
    while True:
        picked_axis0 = []
        picked_axis1 = []
        zerocnt = numpy.concatenate([(w == 0).sum(axis=1), (w == 0).sum(axis=0)], axis=0)

        while zerocnt.max() > 0:

            maxindex = zerocnt.argmax()
            if maxindex < m:
                picked_axis0.append(maxindex)
                zerocnt[numpy.argwhere(w[maxindex, :] == 0).squeeze(1) + m] = \
                    numpy.maximum(zerocnt[numpy.argwhere(w[maxindex, :] == 0).squeeze(1) + m] - 1, 0)
                zerocnt[maxindex] = 0
            else:
                picked_axis1.append(maxindex - m)
                zerocnt[numpy.argwhere(w[:, maxindex - m] == 0).squeeze(1)] = \
                    numpy.maximum(zerocnt[numpy.argwhere(w[:, maxindex - m] == 0).squeeze(1)] - 1, 0)
                zerocnt[maxindex] = 0
        if len(picked_axis0) + len(picked_axis1) < m:
            left_axis0 = list(set(list(range(m))) - set(list(picked_axis0)))
            left_axis1 = list(set(list(range(m))) - set(list(picked_axis1)))
            delta = w[left_axis0, :][:, left_axis1].min()
            w[left_axis0, :] -= delta
            w[:, picked_axis1] += delta
        else:
            break
    pos = []
    for i in range(m):
        pos.append(list(numpy.argwhere(w[i, :] == 0).squeeze(1)))

    def search(layer, path):
        if len(path) == m:
            return path
        else:
            for i in pos[layer]:
                if i not in path:
                    newpath = copy.deepcopy(path)
                    newpath.append(i)
                    ans = search(layer + 1, newpath)
                    if ans is not None:
                        return ans
            return None

    path = search(0, [])
    totalcorrect = 0
    for i, j in enumerate(path):
        totalcorrect += total[i, j]
    return totalcorrect / n


from munkres import Munkres, print_matrix


def best_map(L1, L2):
    """
    http://www.simyng.com/index.php/archives/89/
    Parameters
    ----------
    L1
    L2

    Returns
    -------

    """
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = numpy.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = numpy.unique(L2)
    nClass2 = len(Label2)
    nClass = numpy.maximum(nClass1, nClass2)

    G = numpy.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = numpy.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = numpy.array(index)
    c = index[:, 1]
    newL2 = numpy.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(truth, pred):
    c_x = best_map(truth, pred)
    err_x = numpy.sum(truth[:] != c_x[:])
    missrate = err_x.astype(float) / (truth.shape[0])
    return missrate


def test():
    """

    Returns
    -------

    """
    t = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    p = [1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 0, 0, 0, 4, 1]
    """
    0.8333333333333334
    0.8333333333333334
    0.7333333333333333
    0.7333333333333334
    """

    # t = [2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1]
    # p = [3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2]
    """
    1.0
    1.0
    """

    truth = numpy.array(t)
    pred = numpy.array(p)
    print(hungarian_cluster_acc(truth, pred))
    print(1 - err_rate(truth, pred))


if __name__ == '__main__':
    """"""
    test()
    p = "../../dataSet/data/test.csv"
    samples = pandas.read_csv(p, usecols=[0, 1])
