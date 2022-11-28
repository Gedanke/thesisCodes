# -*- coding: utf-8 -*-

import os
import random
import pandas
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat


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
        new_data = data

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
            '''将第一列列名修改为 x，第一列列名修改为 y'''
            data = data.rename(columns={columns_name[0]: "x", columns_name[1]: "y", columns_name[-1]: "num"})
            '''对数据归一化处理'''
            if self.parameters["norm"] != 0:
                new_data = self.normalized(data[columns_name[0:-1]])
                if len(columns_name) > 2:
                    '''最后一列列名修改为 num'''
                    new_data["num"] = data[columns_name[-1]]
            else:
                new_data = data

            '''保存结果'''
            new_data.to_csv(self.path + "data/demo/" + file_name + ".csv", index=False)

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
            file_path = deal_path + dir_name + "/" + dir_name + ".mat"
            '''加载数据集'''
            mat = loadmat(file_path)
            '''样本数量与特征个数'''
            samples_num, features_num = mat["X"].shape
            '''数据集'''
            data = pandas.DataFrame(mat["X"], columns=list(range(features_num)))
            '''归一化数据'''
            data = self.normalized(data)
            '''标签'''
            data["label"] = [
                _[0] for _ in mat["Y"].tolist()
            ]
            '''保存结果'''
            data.to_csv(self.path + "/data/uci/" + dir_name + "/" + dir_name + ".csv", index=False)

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
            if "gmu" in self.parameters and "sigma" in self.parameters:
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
                noise_data.at[i, j] += random.gauss(self.parameters["gmu"], self.parameters["sigma"])

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
        res += "__n_" + str(self.parameters["norm"])
        res += "__m_" + str(self.parameters["gmu"])
        res += "__s_" + str(self.parameters["sigma"])
        '''加上文件后缀'''
        res += file_type

        return res

    def param_data_mnist(self):
        """
        处理 data 文件夹下的 mnist 文件夹下的数据集
        Returns
        -------

        """
        '''处理 mnist 文件下的数据'''
        deal_path = self.path + "data/mnist/"
        '''该文件夹下的所有文件'''
        files = os.listdir(deal_path)

    def param_data_uci(self):
        """
        处理 data 文件夹下的 uci 文件夹下的数据集
        Returns
        -------

        """
        '''处理 uci 文件下的数据'''
        deal_path = self.path + "data/uci/"
        '''该文件夹下的所有文件'''
        dirs = os.listdir(deal_path)

        for dir_name in dirs:
            '''文件路径'''
            file_path = deal_path + dir_name + "/" + dir_name + ".csv"
            '''读取数据'''
            data = pandas.read_csv(file_path)
            '''保存结果'''
            data.to_csv(self.path + "/experiment/uci/" + dir_name + "/" + dir_name + ".csv", index=False)
