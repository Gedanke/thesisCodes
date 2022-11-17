# -*- coding: utf-8 -*-

import random
import matplotlib.pyplot as plt
from dpc import *
from sklearn.datasets import *

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


def get_path(path, param):
    """
    得到保存结果文件的路径
    Parameters
    ----------
    path: 文件路径
    param: 参数

    Returns
    -------
    res: 文件路径
    """
    '''文件路径，文件全名'''
    dir_path, file = os.path.split(path)
    '''文件名，文件后缀'''
    file_name, file_type = os.path.splitext(file)
    '''在原始文件下建立以该文件名命名的文件夹'''
    res = dir_path + "/" + file_name + "/"
    '''判断该路径是存在'''
    if not os.path.isdir(res):
        '''创建文件夹'''
        os.mkdir(res)

    '''加上文件名'''
    res += file_name
    '''param 字典参数拼接'''
    for k, v in param.items():
        res += "_" + str(k) + "_" + str(v)
    '''加上文件后缀'''
    res += file_type

    return res


def add_noise(data, param):
    """
    https://blog.csdn.net/weixin_46072771/article/details/105726470
    数据添加噪声
    Parameters
    ----------
    data(引用传递): 原始数据，不包含标签列
    param: 参数

    Returns
    -------
    noise_data: 噪声数据
    """
    '''深拷贝一份'''
    noise_data = data.copy()
    '''样本数量'''
    num = len(noise_data)
    '''列名'''
    col = list(noise_data.columns)
    '''添加高斯噪声'''
    for i in range(num):
        for j in col:
            noise_data.at[i, j] += random.gauss(param["mu"], param["sigma"])

    return noise_data


def draw_points(data, plot):
    """
    绘制原始数据分布图
    Parameters
    ----------
    data: 数据
    plot: 绘图句柄

    Returns
    -------

    """
    '''绘制散点图'''
    plot.scatter(data.loc[:, 'x'], data.loc[:, 'y'], c='k')

    '''设置 x 轴'''
    plot.set_xlabel("x")
    '''设置 y 轴'''
    plot.set_ylabel("y")


def get_noise_data(param):
    """
    使用 Sklearn 生成数据集
    https://blog.csdn.net/weixin_46072771/article/details/105726470
    Parameters
    ----------
    param

    Returns
    -------

    """
    '''太极型非凸集样本点'''
    data, label = make_moons(n_samples=1500, shuffle=True, noise=param["noise"], random_state=None)
    plt.scatter(data[:, 0], data[:, 1], c=label, s=7)
    plt.show()


def get_noise_data_():
    """
    使用 Sklearn 生成数据集
    https://blog.csdn.net/weixin_46072771/article/details/105726470
    Parameters
    ----------

    Returns
    -------

    """
    '''正太分布数据'''
    # data, label = make_blobs(n_samples=1500, n_features=2, centers=5)
    # plt.scatter(data[:, 0], data[:, 1], c=label, s=6)
    # plt.show()

    '''同心圆数据'''
    # data, label = make_circles(n_samples=15000, shuffle=True, noise=0.03, random_state=None, factor=0.6)
    # plt.scatter(data[:, 0], data[:, 1], c=label, s=6)
    # plt.show()

    '''模拟分类数据集'''
    data, label = make_classification(n_samples=500, n_features=20, n_informative=2,
                                      n_redundant=2, n_repeated=0, n_classes=2,
                                      n_clusters_per_class=2, weights=None,
                                      flip_y=0.01, class_sep=1.0, hypercube=True,
                                      shift=0.0, scale=1.0, shuffle=True, random_state=None)
    '''共 20 个特征维度，此处仅使用两个维度作图演示'''
    plt.scatter(data[:, 0], data[:, 1], c=label, s=7)
    plt.show()

    '''太极型非凸集样本点'''
    # data, label = make_moons(n_samples=1500, shuffle=True, noise=0.06, random_state=None)
    # plt.scatter(data[:, 0], data[:, 1], c=label, s=7)
    # plt.show()

    '''同心圆形样本点'''
    # data, label = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)
    # plt.scatter(data[:, 0], data[:, 1], marker='o', c=label)
    # plt.show()

    '''二进制分类数据'''
    # data, label = make_hastie_10_2(n_samples=1000)
    # plt.scatter(data[:, 0], data[:, 1], marker='o', c=label)
    # plt.show()

    '''瑞士卷曲线数据集'''
    # data, label = make_swiss_roll(n_samples=2000, noise=0.1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral, edgecolors='black')
    # plt.show()

    ''''''
    # data, label = make_s_curve(n_samples=1000)
    # plt.scatter(data[:, 0], data[:, 1], marker='o', c=label)
    # plt.show()


def add_noise_():
    """

    Returns
    -------

    """
    p = "../../dataSet/data/spiral.csv"
    par = {"mu": 0, "sigma": 0.7}
    pp = get_path(p, par)
    d = pandas.read_csv(p, usecols=[0, 1])

    dd = add_noise(d, par)

    '''两个属性，二维数据能够可视化，做四个图'''
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    '''调整边界'''
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    draw_points(d, axes[0])
    draw_points(dd, axes[1])
    axes[0].set_title("raw data")
    axes[1].set_title("noise data")

    plt.show()


if __name__ == "__main__":
    """"""
    # get_noise_data({"noise": 0.15})
    get_noise_data_()
