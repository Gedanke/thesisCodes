# -*- coding: utf-8 -*-

from deal import *

'''LoadData 类的相关参数'''
param = {
    "norm": 0,
    "mu": 10,
    "sigma": 0.7,
    "make": {
        "samples": 300,
        "features": 12,
        "classes": 4,
        "noise": 0.2,
        "random": 4
    }
}


def run_algorithm():
    """
    运行 IDPC 算法
    Returns
    -------

    """
    '''experiment 下的 build 文件夹'''

    '''experiment 下的 demo 文件夹'''
    run_demo()

    '''experiment 下的 mnist 文件夹'''

    '''experiment 下的 uci 文件夹'''


def run_demo():
    """

    Returns
    -------

    """


def different_dpc():
    """
    不同类型的 dpc 算法
    Returns
    -------

    """


def helper(dir_name):
    """

    Parameters
    ----------
    dir_name

    Returns
    -------

    """


if __name__ == "__main__":
    """"""
    p = "../../dataSet/"

    # ld = LoadData(p, param)
    # ld.param_data_build()
    # ld.param_data_demo()
