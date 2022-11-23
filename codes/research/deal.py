# -*- coding: utf-8 -*-

import random
from dpcp import *
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler

'''自建数据集目录'''

make_data_set = {
    "blobs", "circles", "classification", "moons", "gaussian_quantiles",
    "hastie_10_2", "swiss_roll"
}


class LoadData:
    """
    加载 ../../dataSet/ 下的不同数据集，做出对应的处理
    1、将 raw 中的原始数据集合进行处理，保存到 data 下
    2、将 data 下的数据添加参数、保存到 experiment 下

    demo：一些常见的 UCI 数据，除了 origin 数据集没有标签，其他数据都有标签，均含有两个属性 x 与 y，标签为 num，使用 \t 作为分隔符
    build: 使用 sklearn.datasets 库生成的数据集。样本个数，属性个数，类簇数，随机数种子，噪声级别等等
    """

    def __init__(self, path, parameters=None):
        """
        初始化相关成员
        Parameters
        ----------
        path: 文件路径
        parameters: 构造处理数据集需要的参数
        """
        '''文件路径'''
        self.path = path
        '''构造处理数据集需要的参数'''
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def normalized(self, data):
        """
        数据归一化处理
        Parameters
        ----------
        data: 原始数据

        Returns
        -------
        new_data: 返回处理后的数据
        """
        '''处理后的数据'''
        new_data = None

        if self.parameters["norm"] == 1:
            '''归一化'''
            new_data = (data - data.min()) / (data.max() - data.min())
        elif self.parameters["norm"] == 2:
            '''标准化'''
            preprocess = StandardScaler()
            new_data = preprocess.fit_transform(data)

        return new_data

    def deal_raw_demo(self):
        """
        处理 raw 文件夹下的 demo 文件下的数据
        Returns
        -------

        """
        '''处理 demo 文件下的数据'''
        deal_path = self.path + "raw/demo/"
        '''该文件夹下的所有文件'''
        files = os.listdir(deal_path)

        for file in files:
            '''文件路径'''
            file_path = deal_path + file
            '''文件路径，文件全名'''
            dir_path, file = os.path.split(file_path)
            '''文件名，文件后缀'''
            file_name = os.path.splitext(file)[0]
            '''读取数据'''
            data = pandas.read_csv(file_path, sep="\t")
            '''获取列名'''
            columns_name = list(data.columns)
            '''将第一列列名修改为 x，第一列列名修改为 y，最后一列列名修改为 num'''
            data = data.rename(columns={columns_name[0]: "x", columns_name[1]: "y", columns_name[-1]: "num"})
            '''对数据归一化处理'''
            if self.parameters["norm"] != 0:
                data = self.normalized(data)
            '''保存结果'''
            data.to_csv(self.path + "data/demo/" + file_name + ".csv", index=False)

    def deal_raw_mnist(self):
        """
        处理 raw 文件夹下的 mnist 文件下的数据
        Returns
        -------

        """
        '''处理 mnist 文件下的数据'''
        deal_path = self.path + "raw/mnist/"
        '''该文件夹下的所有文件'''
        files = os.listdir(deal_path)

    def deal_raw_uci(self):
        """
        处理 raw 文件夹下的 uci 文件下的数据
        Returns
        -------

        """
        '''处理 uci 文件下的数据'''
        deal_path = self.path + "raw/uci/"
        '''该文件夹下的所有文件'''
        dirs = os.listdir(deal_path)

        for dir_name in dirs:
            '''文件路径'''
            file_path = deal_path + dir_name + "/" + dir_name + ".data"

    def param_data_demo(self):
        """
        处理 data 文件夹下的 demo 文件夹下的数据集
        Returns
        -------

        """
        '''处理 demo 文件下的数据'''
        deal_path = self.path + "data/demo/"
        '''该文件夹下的所有文件'''
        files = os.listdir(deal_path)

        for file in files:
            '''文件路径'''
            file_path = deal_path + file
            '''保存结果文件路径'''
            save_path = self.get_experiment_path(file, "demo")
            '''读取数据'''
            data = pandas.read_csv(file_path)
            '''判断高斯噪声的两个参数是否存在'''
            if "mu" in self.parameters and "sigma" in self.parameters:
                '''返回增加了噪声的数据'''
                data = self.add_noise(data)
            '''保存结果'''
            data.to_csv(save_path, index=False)

    def add_noise(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------

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
                '''添加高斯噪声'''
                noise_data.at[i, j] += random.gauss(self.parameters["mu"], self.parameters["sigma"])

        return noise_data

    def get_experiment_path(self, file, dir_name):
        """
        得到保存结果文件的路径
        Parameters
        ----------
        file: 文件名
        dir_name: 文件夹名

        Returns
        -------
        res: 文件路径
        """
        '''文件名，文件后缀'''
        file_name, file_type = os.path.splitext(file)

        '''在原始文件下建立以该文件名命名的文件夹'''
        res = self.path + "experiment/" + dir_name + "/" + file_name + "/"
        '''判断该路径是存在'''
        if not os.path.isdir(res):
            '''创建文件夹'''
            os.mkdir(res)

        '''加上文件名'''
        res += file_name
        '''param 字典参数拼接'''
        res += "__norm_" + str(self.parameters["norm"])
        res += "__mu_" + str(self.parameters["mu"])
        res += "__sigma_" + str(self.parameters["sigma"])
        '''加上文件后缀'''
        res += file_type

        return res

    def param_data_build(self):
        """
        自建一些数据集
        shuffle 统一设置成 False
        Returns
        -------

        """
        '''生成符合正态分布数据'''
        data, label = make_blobs(n_samples=self.parameters["make"]["samples"],
                                 n_features=self.parameters["make"]["features"],
                                 centers=self.parameters["make"]["classes"], shuffle=False,
                                 random_state=self.parameters["make"]["random"])
        '''保存结果'''
        self.save_param_data(data, label, "blobs",
                             "__s_" + str(self.parameters["make"]["samples"]) +
                             "__f_" + str(self.parameters["make"]["features"]) +
                             "__c_" + str(self.parameters["make"]["classes"]) +
                             "__r_" + str(self.parameters["make"]["random"]))

        '''生成同心圆样本点'''
        data, label = make_circles(n_samples=self.parameters["make"]["samples"], shuffle=False,
                                   noise=self.parameters["make"]["noise"],
                                   random_state=self.parameters["make"]["random"])

        '''保存结果'''
        self.save_param_data(data, label, "circles",
                             "__s_" + str(self.parameters["make"]["samples"]) +
                             "__n_" + str(self.parameters["make"]["noise"]) +
                             "__r_" + str(self.parameters["make"]["random"]))

        '''生成太极型非凸集样本点'''
        data, label = make_moons(n_samples=self.parameters["make"]["samples"], shuffle=False,
                                 noise=self.parameters["make"]["noise"],
                                 random_state=self.parameters["make"]["random"])
        '''保存结果'''
        self.save_param_data(data, label, "moons",
                             "__s_" + str(self.parameters["make"]["samples"]) +
                             "__n_" + str(self.parameters["make"]["noise"]) +
                             "__r_" + str(self.parameters["make"]["random"]))

        '''生成同心圆形样本点'''
        data, label = make_gaussian_quantiles(n_samples=self.parameters["make"]["samples"],
                                              n_features=self.parameters["make"]["features"],
                                              n_classes=self.parameters["make"]["classes"], shuffle=False,
                                              random_state=self.parameters["make"]["random"])
        '''保存结果'''
        self.save_param_data(data, label, "gaussian_quantiles",
                             "__s_" + str(self.parameters["make"]["samples"]) +
                             "__f_" + str(self.parameters["make"]["features"]) +
                             "__c_" + str(self.parameters["make"]["classes"]) +
                             "__r_" + str(self.parameters["make"]["random"]))

        '''生成二进制分类数据'''
        data, label = make_hastie_10_2(n_samples=self.parameters["make"]["samples"],
                                       random_state=self.parameters["make"]["random"])
        '''保存结果'''
        self.save_param_data(data, label, "hastie_10_2",
                             "__s_" + str(self.parameters["make"]["samples"]) +
                             "__r_" + str(self.parameters["make"]["random"]))

    def save_param_data(self, data, label, dir_name, suffix):
        """
        合并 data 和 label 为一个 csv 文件并写入到指定文件夹下
        Parameters
        ----------
        data: 数据集
        label: 标签列
        dir_name: 文件名
        suffix: 后缀

        Returns
        -------

        """
        '''文件路径'''
        path = self.path + "experiment/build/" + dir_name + "/"
        '''判断路径是否存在'''
        if not os.path.isdir(path):
            '''创建文件夹'''
            os.mkdir(path)

        '''文件名拼接'''
        path += dir_name + suffix + ".csv"

        '''data 保存为 pandas.DataFrame'''
        col = list(range(self.parameters["make"]["features"]))

        if dir_name in {"circles", "moons"}:
            '''只有两个属性'''
            col = list(range(2))
        elif dir_name == "hastie_10_2":
            '''只有十个属性'''
            col = list(range(10))

        data = pandas.DataFrame(data, columns=col)
        '''添加标签列'''
        data["num"] = label

        '''保存数据'''
        data.to_csv(path, index=False)

    def param_data_mnist(self):
        """
        处理 data 文件夹下的 mnist 文件夹下的数据集
        Returns
        -------

        """

    def param_data_uci(self):
        """
        处理 data 文件夹下的 uci 文件夹下的数据集
        Returns
        -------

        """


