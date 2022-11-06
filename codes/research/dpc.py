# -*- coding: utf-8 -*-

import os
import math
import json
import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.metrics import *

SAVE_PATH = "../../results/"


class DPC:
    """
    经典 DPC 算法基类：
    加载数据(统一为 csv 格式，第一列列名为 x，第一列列名为 y，若存在标签，最后一列为标签)
    计算距离矩阵
    计算截断距离、计算局部密度
    计算相对距离、确定聚类中心
    样本分配，得到聚类结果(图表)

    部分方法(但不是抽象方法)会在不同子类中重新继承实现，对应学位论文中不同的创新点
    即学位论文中三个方向会对应着不同的子类
    """

    def __init__(self, path, use_cols=None, num=0, dc_method=0, dc_percent=1, rho_method=1, delta_method=1,
                 use_halo=False, plot=None):
        """
        初始化成员变量
        Args:
            path: 文件完整路径
            use_cols: 使用的列，仅有两个属性的数据集才可以做出数据原始结构图和聚类结果图
            num: 聚类类簇数
            dc_method: 截断距离计算方法
            dc_percent: 截断距离百分比数
            rho_method: 局部密度计算方法
            delta_method: 相对距离计算方法
            use_halo: 是否计算光晕点
            plot: 绘图句柄
        """
        '''构造函数中的相关参数'''
        '''文件完整路径'''
        self.path = path
        '''从文件路径中获取文件名(不含后缀)'''
        self.data_name = os.path.splitext(os.path.split(self.path)[-1])[0]
        '''使用的列。两列，做四个结果图；否则做两个图'''
        if use_cols is None:
            use_cols = [0, 1]
        self.use_cols = use_cols
        '''聚类类簇数，可以指定，也可以从文件中读取得到(默认从最后一列中读取)'''
        self.num = num
        '''截断距离计算方法'''
        self.dc_method = dc_method
        '''截断距离百分比数'''
        self.dc_percent = dc_percent
        '''局部密度计算方法'''
        self.rho_method = rho_method
        '''相对距离计算方法'''
        self.delta_method = delta_method
        '''是否计算光晕点'''
        self.use_halo = use_halo
        '''绘图句柄'''
        self.plot = plot

        '''其他参数'''
        '''边界域中密度最大的点'''
        self.border_b = list()
        '''数据集的所有样本点，不包括标签列'''
        self.samples = pandas.DataFrame({})
        '''样本个数'''
        self.samples_num = 0
        '''是否含有标签列'''
        self.label_sign = True
        '''真实标签'''
        self.label_true = list()
        '''聚类结果'''
        self.label_pred = list()
        '''距离矩阵'''
        self.dis_matrix = pandas.DataFrame({})
        '''外部指标，不需要列标签'''
        self.cluster_result_unsupervised = {
            "davies_bouldin": 0.0,
            "calinski_harabasz": 0.0,
            "silhouette_coefficient": 0.0
        }
        '''内部指标，需要列标签'''
        self.cluster_result_supervised = dict()

    def cluster(self):
        """
        运行算法
        Returns:

        """
        '''获取数据集相关信息。如样本点，距离矩阵，欧式距离，最小距离，最大距离，样本数'''
        dis_array, min_dis, max_dis = self.load_points_msg()
        '''计算截断距离 dc'''
        dc = self.get_dc(dis_array, min_dis, max_dis)
        '''计算局部密度 rho'''
        rho = self.get_rho(dc)
        '''计算相对距离 delta'''
        delta = self.get_delta(rho)
        '''确定聚类中心，计算 gamma(局部密度于相对距离的乘积)'''
        center, gamma = self.get_center(rho, delta)
        '''非聚类中心样本点分配'''
        cluster_result = self.assign(rho, center)
        '''光晕点'''
        halo = list()
        if self.use_halo:
            cluster_result, halo = self.get_halo(rho, cluster_result, dc)

        '''绘图结果'''
        self.show_plot(rho, delta, center, cluster_result, halo)
        '''得到聚类标签'''
        self.label_pred = [-1 for _ in range(self.samples_num)]
        for c, points in cluster_result.items():
            for point in points:
                self.label_pred[point] = c
        '''聚类结果'''
        self.show_result()

    def distance(self, metric='euclidean'):
        """
        (重写方法)
        样本度量方法
        后续如果改进样本间相似性度量，重写该方法
        返回的依然是距离度量的矩阵(列表)
        维度为 self.samples_num * (self.samples_num - 1) / 2
        Parameters
        ----------
        metric: 距离度量方法

        Returns
        -------
        dis_array: 距离度量矩阵(列表)
        """
        '''此处使用的是欧式距离'''
        dis_array = sch.distance.pdist(self.samples, metric)
        return dis_array

    def load_points_msg(self):
        """
        获取数据集相关信息。如样本点，距离矩阵，欧式距离，最小距离，最大距离，样本数

        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        '''读取 csv 文件全部内容'''
        self.samples = pandas.read_csv(self.path)
        '''数据的列数目是否等于给定的'''
        col = list(self.samples.columns)
        if len(col) == len(self.use_cols):
            '''等于，说明没有标签列，self.label_sign 置为 False'''
            self.label_sign = False
        else:
            '''否则，说明存在标签列，self.label_sign 置为 True'''
            # self.label_sign = True
            '''获取真实标签列，存放到 self.label_true 中'''
            self.label_true = self.samples[col[-1]].tolist()
            '''初始化内部评价指标'''
            self.cluster_result_supervised = {
                "rand_index": 0.0,
                "adjusted_rand_index": 0.0,
                "mutual_info": 0.0,
                "normalized_mutual_info": 0.0,
                "adjusted_mutual_info": 0.0,
                "homogeneity": 0.0,
                "completeness": 0.0,
                "v_measure": 0.0,
                "homogeneity_completeness_v_measure": 0.0,
                "fowlkes-mallows_index": 0.0
            }

        '''无论是否相等，都对 self.samples 进行切片，列为 self.use_cols'''
        self.samples = self.samples.iloc[:, self.use_cols]
        '''得到样本数目'''
        self.samples_num = len(self.samples)

        '''距离矩阵初始化'''
        self.dis_matrix = pandas.DataFrame(numpy.zeros((self.samples_num, self.samples_num)))
        '''维度为 self.samples_num * (self.samples_num - 1) / 2'''
        dis_array = self.distance('euclidean')

        '''对距离矩阵的处理'''
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                '''赋值'''
                self.dis_matrix.at[i, j] = dis_array[num]
                '''处理对角元素'''
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        '''最小距离'''
        min_dis = self.dis_matrix.min().min()
        '''最大距离'''
        max_dis = self.dis_matrix.max().max()

        return dis_array, min_dis, max_dis

    def get_dc(self, dis_array, min_dis, max_dis):
        """
        计算局部密度
        Parameters
        ----------
        dis_array: self.dis_matrix 的上三角距离矩阵(一维)
        min_dis: 最小距离
        max_dis: 最大距离

        Returns
        -------
        dc: 截断距离
        """
        lower = self.dc_percent / 100
        upper = (self.dc_percent + 1) / 100
        '''判断计算截断距离的方法，默认是该方法'''
        if self.dc_method == 0:
            while 1:
                dc = (min_dis + max_dis) / 2
                '''上三角矩阵'''
                neighbors_percent = len(dis_array[dis_array < dc]) / (((self.samples_num - 1) ** 2) / 2)
                if lower <= neighbors_percent <= upper:
                    return dc
                elif neighbors_percent > upper:
                    max_dis = dc
                elif neighbors_percent < lower:
                    min_dis = dc
        '''如果对截断距离计算有所改进，可以直接重写该方法或者添加到 if 语句中'''

    def get_rho(self, dc):
        """
        (重写方法)
        计算局部密度
        Parameters
        ----------
        dc: 截断距离

        Returns
        -------
        rho: 每个样本点的局部密度
        """
        rho = numpy.zeros(self.samples_num)

        for i in range(self.samples_num):
            if self.rho_method == 0:
                '''到样本点 i 距离小于 dc 的点数量'''
                rho[i] = len(self.dis_matrix.loc[i, :][self.dis_matrix.loc[i, :] < dc]) - 1
            elif self.rho_method == 1:
                '''高斯核'''
                for j in range(self.samples_num):
                    if i != j:
                        rho[i] += math.exp(-(self.dis_matrix.at[i, j] / dc) ** 2)
            elif self.rho_method == 2:
                '''排除异常值'''
                n = int(self.samples_num * 0.05)
                '''选择前 n 个离 i 最近的样本点'''
                rho[i] = math.exp(-(self.dis_matrix.loc[i].sort_values().values[:n].sum() / (n - 1)))

        return rho

    def get_delta(self, rho):
        """
        计算相对距离
        Parameters
        ----------
        rho: 局部密度

        Returns
        -------
        delta: 相对距离
        """
        delta = numpy.zeros(self.samples_num)

        '''考虑局部密度 rho 是否存在多个最大值'''
        if self.delta_method == 0:
            '''不考虑 rho 相同且同时为最大的情况'''
            for i in range(self.samples_num):
                '''第 i 个样本的局部密度'''
                rho_i = rho[i]
                '''局部密度比 rho_i 大的点集合的索引'''
                j_list = numpy.where(rho > rho_i)[0]
                if len(j_list) == 0:
                    '''rho_i 是 rho 中局部密度最大的点'''
                    delta[i] = self.dis_matrix.loc[i, :].max()
                else:
                    '''非局部密度最大的点，寻找局部密度大于 i 且离 i 最近的点的索引'''
                    min_dis_idx = self.dis_matrix.loc[i, j_list].idxmin()
                    delta[i] = self.dis_matrix.at[i, min_dis_idx]
        elif self.delta_method == 1:
            '''考虑 rho 相同且同时为最大的情况(推荐)'''
            '''将局部密度从大到小排序，返回对应的索引值'''
            rho_order_idx = rho.argsort()[-1::-1]
            '''即 rho_order_idx 存放的是第 i 大元素 rho 的索引，而不是对应的值'''
            for i in range(1, self.samples_num):
                '''第 i 个样本'''
                rho_idx = rho_order_idx[i]
                '''j < i 的排序索引值或者说 rho > i 的索引列表'''
                j_list = rho_order_idx[:i]
                '''返回第 idx_rho 行，j_list 列中最小值对应的索引'''
                min_dis_idx = self.dis_matrix.loc[rho_idx, j_list].idxmin()
                '''i 的相对距离，即比 i 大，且离 i 最近的距离'''
                delta[rho_idx] = self.dis_matrix.at[rho_idx, min_dis_idx]
            '''相对距离最大的点'''
            delta[rho_order_idx[0]] = delta.max()

        return delta

    def get_center(self, rho, delta):
        """
        (重写方法)
        获取聚类中心，计算 gamma
        Parameters
        ----------
        rho: 局部密度
        delta: 相对距离

        Returns
        -------
        center: 距离中心列表
        gamma: rho * delta
        """
        center = None
        gamma = rho * delta
        '''对 gamma 排序'''
        gamma = pandas.DataFrame(gamma, columns=["gamma"]).sort_values("gamma", ascending=False)
        if self.num > 0:
            '''取 gamma 中前 self.samples_num 个点作为聚类中心'''
            center = numpy.array(gamma.index)[:self.num]
        else:
            '''采用其他方法，这里是未来重写的重点'''
            # center = gamma[gamma.gamma > threshold].loc[:, "gamma"].index

        return center, gamma

    def assign(self, rho, center):
        """
        (重写方法)
        非聚类中心样本点分配
        Parameters
        ----------
        rho: 局部密度
        center: 聚类中心样本点

        Returns
        -------
        cluster: 聚类结果，dict(center: str, points: list())
        """
        '''链式分配方法(顺藤摸瓜)'''
        cluster_result = dict()
        '''键为聚类中心索引，值为归属聚类中心的所有样本点，包括聚类中心本身'''
        for c in center:
            cluster_result[c] = list()

        '''link 的键为当前样本，值为离当前点最近的样本点'''
        link = dict()
        '''局部密度从大到小排序，返回索引值'''
        order_rho_idx = rho.argsort()[-1::-1]
        for i, v in enumerate(order_rho_idx):
            if v in center:
                '''聚类中心'''
                link[v] = v
                continue
            '''非聚类中心的点'''
            '''前 i 个局部密度点的排序索引值，也就是局部密度大于 rho[v] 的索引列表'''
            rho_idx = order_rho_idx[:i]
            '''在局部密度大于 rho[v] 的点中，距离从小到大排序的第一个索引，也是离得最近的点(不一定是聚类中心)'''
            link[v] = self.dis_matrix.loc[v, rho_idx].sort_values().index.tolist()[0]

        '''分配所有样本点'''
        for k, v in link.items():
            '''使用 c 纪录离 k 最近的点 v'''
            c = v
            '''c 不是聚类中心'''
            while c not in center:
                '''c 更新为离 c 最近的点 link[c]，一步步迭代，顺藤摸瓜，直到找到 c 对应的聚类中心'''
                c = link[c]
            '''c 是聚类中心，分配当前 k 到 c 中'''
            cluster_result[c].append(k)

        '''最近中心分配方法'''
        """
        for i in range(self.samples_num):
            c = self.dis_matrix.loc[i, center].idxmin()
            cluster_result[c].append(i)
        """

        return cluster_result

    def get_halo(self, rho, cluster_result, dc):
        """
        获取光晕点
        Parameters
        ----------
        rho: 局部密度
        cluster_result: 聚类结果
        dc: 截断距离

        Returns
        -------
        cluster_result: 聚类结果
        halo: 光晕点
        """
        '''所有样本点'''
        all_points = set(list(range(self.samples_num)))
        for c, points in cluster_result.items():
            '''属于其他聚类的点'''
            others_points = list(set(all_points) - set(points))
            border = list()
            for point in points:
                '''到其他聚类中心点的距离小于 dc'''
                if self.dis_matrix.loc[point, others_points].min() < dc:
                    border.append(point)
            if len(border) == 0:
                continue
            '''边界域中密度最大的值'''
            # rbo_b = rho[border].max()
            '''边界域中密度最大的点'''
            point_b = border[rho[border].argmax()]
            self.border_b.append(point_b)
            '''边界域最大密度'''
            rho_b = rho[point_b]
            '''筛选可靠性高的点'''
            filter_points = numpy.where(rho >= rho_b)[0]
            '''该聚类中可靠性高的点'''
            points = list(set(filter_points) & set(points))
            cluster_result[c] = points

        '''halo'''
        cluster_points = set()
        for c, points in cluster_result.items():
            cluster_points = cluster_points | set(points)
        '''光晕点'''
        halo = list(set(all_points) - cluster_points)

        return cluster_result, halo

    def show_plot(self, rho, delta, center, cluster_result, halo):
        """
        绘图
        Parameters
        ----------
        rho
        delta
        center
        cluster_result
        halo

        Returns
        -------

        """
        if self.plot is None:
            '''未指定绘图句柄，单一数据绘图'''
            if len(self.use_cols) == 2:
                '''两个属性，二维数据能够可视化，做四个图'''
                fig, axes = plt.subplots(2, 2, figsize=(18, 18))
                '''调整边界'''
                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                '''第一张图，原始数据分布图'''
                self.draw_points([], axes[0][0])
                '''第二张图，rho 与 delta 的二维决策图'''
                self.draw_rho_delta(rho, delta, center, axes[0][1])
                '''第三张图，gamma'''
                self.draw_gamma(rho * delta, axes[1][0])
                '''第四张图，聚类结果图'''
                self.draw_cluster(cluster_result, halo, axes[1][1])
                '''保存图片'''
                plt.savefig(SAVE_PATH + "plot/" + self.data_name + ".svg")
                plt.show()
            else:
                '''多个属性，多维数据不能直接可视化，做两个图即可'''
                fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                '''调整边界'''
                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                '''第一张图，rho 与 delta 的二维决策图'''
                self.draw_rho_delta(rho, delta, center, axes[0])
                '''第二张图，gamma'''
                self.draw_gamma(rho * delta, axes[1])
                '''保存图片'''
                plt.savefig(SAVE_PATH + "plot/" + self.data_name + ".svg")
                plt.show()
        else:
            '''指定了绘图句柄，多个数据集绘图，只绘制聚类结果图'''
            self.draw_cluster(cluster_result, halo, self.plot)

    def draw_points(self, center, plot):
        """
        绘制原始数据分布图
        Parameters
        ----------
        center: 聚类中心
        plot: 绘图句柄

        Returns
        -------

        """
        plot.scatter(self.samples.loc[:, 'x'], self.samples.loc[:, 'y'], c='k')
        if len(center) != 0:
            '''给了聚类中心，追加并突出聚类中心'''
            center_points = self.samples.loc[center, :]
            plot.scatter(center_points.loc[:, 'x'], center_points.loc[:, 'y'], s=numpy.pi * 8 ** 2)

        '''设置 x 轴'''
        plot.set_xlabel("x")
        '''设置 y 轴'''
        plot.set_ylabel("y")
        '''确定标题'''
        plot.set_title("raw data")

    def draw_rho_delta(self, rho, delta, center, plot):
        """
        绘制局部密度与相对距离的二维决策图
        Parameters
        ----------
        rho: 局部密度
        delta: 相对距离
        center: 聚类中心
        plot: 绘图句柄

        Returns
        -------

        """
        plot.scatter(rho, delta, label="rho-delta", c='k', s=5)
        '''聚类中心点的局部密度'''
        center_rho = rho[center]
        '''聚类中心点的相对距离'''
        center_delta = delta[center]
        '''随机数种子，设置颜色'''
        '''由于该类中确定聚类中心的方法是指定了聚类数，因此使用的是 self.num'''
        if self.num != 0:
            colors = numpy.random.rand(self.num, 3)
        else:
            colors = numpy.random.rand(len(center), 3)
        '''追加并突出聚类中心点的局部密度与相对距离位置'''
        plot.scatter(center_rho, center_delta, c=colors)

        '''设置 x 轴'''
        plot.set_xlabel("rho")
        '''设置 y 轴'''
        plot.set_ylabel("delta")
        '''确定标题'''
        plot.set_title("rho - delta")
        '''图例放在左上角'''
        plot.legend(loc=2)

    def draw_gamma(self, gamma, plot):
        """

        Parameters
        ----------
        gamma: 局部密度与相对距离的乘积
        plot: 绘图句柄

        Returns
        -------

        """
        '''将 gamma 转化为 pandas.DataFrame，并且降序排列'''
        rho_delta = pandas.DataFrame(gamma, columns=["gamma"]).sort_values("gamma", ascending=False)
        plot.scatter(range(self.samples_num), rho_delta.loc[:, "gamma"], label="gamma", s=5)
        # plot.hlines(avg, 0, len(rho_delta), "b", "dashed")

        '''设置 x 轴'''
        plot.set_xlabel("num")
        '''设置 y 轴'''
        plot.set_ylabel("gamma")
        '''确定标题'''
        plot.set_title("gamma(rho * delta)")
        plot.legend(loc="best")

    def draw_cluster(self, cluster_result, halo, plot):
        """
        绘制聚类结果图
        Parameters
        ----------
        cluster_result: 聚类结果
        halo: 光晕点
        plot: 绘图句柄

        Returns
        -------

        """
        cluster_points = dict()
        colors = dict()
        numpy.random.seed(10)

        for k, v in cluster_result.items():
            '''同一类中的样本点'''
            cluster_points[k] = self.samples.loc[cluster_result[k], :]
            colors[k] = numpy.random.rand(3).reshape(1, -1)

        '''绘图'''
        for k, v in cluster_points.items():
            plot.scatter(v.loc[:, "x"], v.loc[:, "y"], c=colors[k], alpha=0.5)
            plot.scatter(v.at[k, "x"], v.at[k, "y"], c=colors[k], s=numpy.pi * 8 ** 2)

        '''光晕点绘制'''
        if len(halo) != 0:
            '''光晕点'''
            noise_pointer = self.samples.loc[halo, :]
            plot.scatter(noise_pointer.loc[:, "x"], noise_pointer.loc[:, "y"], c="k")
            border_b = self.samples.loc[self.border_b, :]
            plot.scatter(border_b.loc[:, "x"], border_b.loc[:, "y"], c="k", s=numpy.pi * 4 ** 2)

        '''设置 x 轴'''
        plot.set_xlabel("x")
        '''设置 y 轴'''
        plot.set_ylabel("y")
        '''确定标题'''
        plot.set_title(self.data_name)

    def show_result(self):
        """
        得出聚类结果
        Returns
        -------

        """
        '''保存所有结果的字典'''
        save_data = dict()

        '''先计算外部评价指标，即不需要标签的'''
        '''戴维森堡丁指数'''
        self.cluster_result_unsupervised["davies_bouldin"] = davies_bouldin_score(self.samples, self.label_pred)
        print("戴维森堡丁指数为：" + str(self.cluster_result_unsupervised["davies_bouldin"]))
        '''CH 分数'''
        self.cluster_result_unsupervised["calinski_harabasz"] = calinski_harabasz_score(self.samples, self.label_pred)
        print("CH 分数为：" + str(self.cluster_result_unsupervised["calinski_harabasz"]))
        '''轮廓系数'''
        self.cluster_result_unsupervised["silhouette_coefficient"] = silhouette_score(self.samples, self.label_pred)
        print("轮廓系数为：" + str(self.cluster_result_unsupervised["silhouette_coefficient"]))

        '''深拷贝，复制一份'''
        save_data = self.cluster_result_unsupervised.copy()

        '''考虑内部评价指标，需要标签的'''
        if self.label_sign:
            '''兰德指数'''
            self.cluster_result_supervised["rand_index"] = rand_score(self.label_true, self.label_pred)
            print("兰德系数为：" + str(self.cluster_result_supervised["rand_index"]))
            '''调整兰德指数'''
            self.cluster_result_supervised["adjusted_rand_index"] = adjusted_rand_score(self.label_true,
                                                                                        self.label_pred)
            print("调整兰德指数为：" + str(self.cluster_result_supervised["adjusted_rand_index"]))
            '''互信息'''
            self.cluster_result_supervised["mutual_info"] = mutual_info_score(self.label_true, self.label_pred)
            print("互信息为：" + str(self.cluster_result_supervised["mutual_info"]))
            '''标准化的互信息'''
            self.cluster_result_supervised["normalized_mutual_info"] = normalized_mutual_info_score(self.label_true,
                                                                                                    self.label_pred)
            print("标准化的互信息为：" + str(self.cluster_result_supervised["normalized_mutual_info"]))
            '''调整互信息'''
            self.cluster_result_supervised["adjusted_mutual_info"] = adjusted_mutual_info_score(self.label_true,
                                                                                                self.label_pred)
            print("调整互信息为：" + str(self.cluster_result_supervised["adjusted_mutual_info"]))
            '''同质性'''
            self.cluster_result_supervised["homogeneity"] = homogeneity_score(self.label_true, self.label_pred)
            print("同质性为：" + str(self.cluster_result_supervised["homogeneity"]))
            '''完整性'''
            self.cluster_result_supervised["completeness"] = completeness_score(self.label_true, self.label_pred)
            print("完整性为：" + str(self.cluster_result_supervised["completeness"]))
            '''调和平均'''
            self.cluster_result_supervised["v_measure"] = v_measure_score(self.label_true, self.label_pred)
            print("调和平均为：" + str(self.cluster_result_supervised["v_measure"]))
            '''融合了同质性、完整性、调和平均'''
            self.cluster_result_supervised["homogeneity_completeness_v_measure"] = homogeneity_completeness_v_measure(
                self.label_true, self.label_pred)
            print("融合了同质性、完整性、调和平均为：" + str(
                self.cluster_result_supervised["homogeneity_completeness_v_measure"]))
            '''Fowlkes-Mallows index'''
            self.cluster_result_supervised["fowlkes-mallows_index"] = fowlkes_mallows_score(self.label_true,
                                                                                            self.label_pred)
            print("Fowlkes-Mallows 指数为：" + str(self.cluster_result_supervised["fowlkes-mallows_index"]))
            '''合并内部与外部评价指标'''
            save_data.update(self.cluster_result_supervised)

        '''所有结果并保存到 json 文件中'''
        with open(SAVE_PATH + "result/" + self.data_name + ".json", "w", encoding="utf-8") as f:
            f.write(json.dumps(save_data, ensure_ascii=False))
