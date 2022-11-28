# -*- coding: utf-8 -*-

from deal import *
from dpc import *
from dpcp import *
from multiprocessing import Pool


def run_algorithm(path, save_path="../../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
                  rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False, plot=None):
    """
    进程函数
    Returns
    -------

    """
    '''运行算法'''
    print(path, params)
    al = DPCD(path, save_path, use_cols, num, dc_method, dc_percent,
              rho_method, delta_method, distance_method, params, use_halo, plot)
    al.cluster()


def run_algorithm_i(path, save_path="../../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
                    rho_method=1, delta_method=1, distance_method='irod', params=None, use_halo=False, plot=None):
    """
    进程函数
    Returns
    -------

    """
    '''运行算法'''
    print(path, params)
    al = IDPC(path, save_path, use_cols, num, dc_method, dc_percent,
              rho_method, delta_method, distance_method, params, use_halo, plot)
    al.cluster()


if __name__ == '__main__':
    """"""
    p = "../../dataSet/"
    param = {
        "norm": 1,
        "gmu": 0,
        "sigma": 0,
        "mu": 10,
        "k": 5
    }
    # ld = LoadData(p, param)
    # ld.deal_raw_demo()
    # ld.deal_raw_uci()
    # ld.param_data_uci()
    # ld.param_data_demo()

    ''''../../dataSet/experiment/uci/USPS/USPS.csv', "../../results/uci/'''
    dp = DPC(
        '../../dataSet/experiment/demo/spiral/spiral__n_1__m_0__s_0.csv', "../../results/demo/", list(range(2)), 3,
        1, 2, 1, 1, 'euclidean', False, None
    )
    dp.cluster()

