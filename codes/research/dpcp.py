# -*- coding: utf-8 -*-

from dpc import *

'''rod 系列度量方法'''
rod_method_set = {
    "rod", "krod", "irod"
}


class DPCD(DPC):
    """
    改进了度量方式的 DPC 算法
    主要改进点集中在 self.load_points_msg 函数内
    并增加了一些其他度量方式进行对比、其他方法基本上保存一致
    """

    def __init__(self, path, save_path="../../results/", use_cols=None, num=0, dc_method=1, dc_percent=1,
                 rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False, plot=None):
        """
        初始化相关成员
        Parameters
        ----------
        path: 文件完整路径，csv 文件，如果是图片，也统一转化为 csv 储存像素点
        use_cols: 使用的列，两个属性(不含标签列)指定与否都可以；超过两列则需要指定读取的列(是否包含标签列)
        num: 聚类类簇数
        dc_method: 截断距离计算方法
        dc_percent: 截断距离百分比数
        rho_method: 局部密度计算方法
        delta_method: 相对距离计算方法
        distance_method: 距离度量方式
        params: 参数集合
        use_halo: 是否计算光晕点
        plot: 绘图句柄，仅有两个属性的数据集才可以做出数据原始结构图和聚类结果图
        """
        super(DPCD, self).__init__(path, save_path=save_path, use_cols=use_cols, num=num, dc_method=dc_method,
                                   dc_percent=dc_percent, rho_method=rho_method, delta_method=delta_method,
                                   distance_method=distance_method, use_halo=use_halo, plot=plot)
        '''其他参数'''
        if params is None:
            params = {}
        self.params = params

    def load_points_msg(self):
        """
        获取数据集相关信息，距离矩阵，ROD 距离列表，最小距离，最大距离
        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        if self.distance_method in metric_way_set:
            return self.distance_standard()
        elif self.distance_method in rod_method_set:
            return self.distance_rods()

    def distance_standard(self):
        """
        使用 sch.distance.pdist 中提供的方法
        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        '''维度为 self.samples_num * (self.samples_num - 1) / 2'''
        dis_array = sch.distance.pdist(self.samples, self.distance_method)

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

    def distance_rods(self):
        """
        ROD 系列度量方式
        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        '''先根据欧式距离生成所有样本的距离列表'''
        dis_array = sch.distance.pdist(self.samples, "euclidean")
        '''这里采用的方法是先深拷贝一份 self.dis_matrix'''
        euclidean_table = self.dis_matrix.copy()

        '''储存两个样本间的欧式距离'''
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                '''赋值'''
                euclidean_table.at[i, j] = dis_array[num]
                '''处理对角元素'''
                euclidean_table.at[j, i] = euclidean_table.at[i, j]
                num += 1

        '''对 euclidean_table 使用 argsort()，该函数会对矩阵的每一行从小到大排序，返回的是 euclidean_table 中索引'''
        rank_order_table = numpy.array(euclidean_table).argsort()

        '''对距离矩阵进行处理'''
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                '''用新的度量方式得到的样本间距离覆盖掉 dis_array'''
                dis_array[num] = self.rods_fun(i, j, rank_order_table, euclidean_table)
                '''self.dis_matrix 内存放样本间的距离'''
                self.dis_matrix.at[i, j] = dis_array[num]
                '''处理对角元素'''
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        '''最小距离'''
        min_dis = self.dis_matrix.min().min()
        '''最大距离'''
        max_dis = self.dis_matrix.max().max()

        return dis_array, min_dis, max_dis

    def rods_fun(self, i, j, rank_order_table, euclidean_table):
        """
        rod 及其改进算法
        Parameters
        ----------
        i: 第 i 个样本
        j: 第 j 个样本
        rank_order_table: 排序距离表
        euclidean_table: 欧式距离表

        Returns
        -------
        res: x1 与 x2 之间的距离
        """
        '''第 i 个样本'''
        x1 = rank_order_table[i, :]
        '''第 j 个样本'''
        x2 = rank_order_table[j, :]
        '''x1 与 x2 之间的距离'''
        res = -1

        '''rod 及其改进算法需要的一些前置条件'''
        '''x1 样本的 id'''
        id_x1 = x1[0]
        '''x2 样本的 id'''
        id_x2 = x2[0]
        '''o_a_b，b 在 a 的排序列表中的索引位置'''
        o_a_b = numpy.where(x1 == id_x2)[0][0]
        '''o_b_a，a 在 b 的排序列表中的索引位置'''
        o_b_a = numpy.where(x2 == id_x1)[0][0]

        '''判断度量方法'''
        if self.distance_method == "rod":
            '''d_a_b，在 a 的排序列表中，从 a 到 b 之间的所有元素在 b 的排序列表中的序数或者索引之和'''
            '''先切片，a 的排序列表中 [a, b] 的索引列表'''
            slice_a_b = x1[0:o_a_b + 1]
            '''索引切片在 b 中的位置序数之和'''
            d_a_b = sum(numpy.where(x2 == slice_a_b[:, None])[-1])
            '''d_b_a，在 b 的排序列表中，从 b 到 a 之间的所有元素在 a 的排序列表中的序数或者索引之和'''
            '''先切片，b 的排序列表中 [b, a] 的索引列表'''
            slice_b_a = x2[0:o_b_a + 1]
            '''索引切片在 a 中的位置序数之和'''
            d_b_a = sum(numpy.where(x1 == slice_b_a[:, None])[-1])
            '''rod'''
            res = (d_a_b + d_b_a) / min(o_a_b, o_b_a)
        elif self.distance_method == "krod":
            '''改进 rod'''
            l_a_b = o_a_b + o_b_a
            '''高斯核'''
            k_a_b = math.exp(-(euclidean_table.at[i, j] / self.params["mu"]) ** 2)
            '''krod，改进 rod 与高斯核相结合'''
            res = l_a_b / k_a_b
        elif self.distance_method == "irod":
            '''改进 rod'''
            l_a_b = (o_a_b + o_b_a) / (len(self.use_cols) - 1)
            '''高斯核'''
            k_a_b = math.exp(-(euclidean_table.at[i, j] / self.params["mu"]) ** 2)
            '''krod，改进 rod 与高斯核相结合'''
            res = l_a_b / k_a_b

        return res

    def get_file_path(self, dir_type):
        """
        保存文件路径
        Parameters
        ----------
        dir_type: 要保存的文件夹(data，plot，result)

        Returns
        -------
        path: 保存结果文件的路径
        """
        '''保存文件路径'''
        path = self.save_path + dir_type + "/" + self.data_name + "_" + self.distance_method + "/"

        '''判断文件夹是否存在'''
        if not os.path.isdir(path):
            '''创建文件夹'''
            os.mkdir(path)

        '''由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数'''
        '''删去 self.dc_method，因为在这个类里面，截断距离都是通过 self.dc_percent 指定的'''
        path += "dcp_" + str(self.dc_percent) + "__rho_" + str(self.rho_method) + \
                "__dem_" + str(self.delta_method) + "__ush_" + str(int(self.use_halo))

        '''判断度量方法'''
        if self.distance_method == 'krod' or self.distance_method == 'irod':
            '''krod、irod 需要参数 mu'''
            path += "__mu_" + str(self.params["mu"])

        '''根据不同的文件夹类型保存不同类型的文件'''
        if dir_type == "data":
            path += ".csv"
        elif dir_type == "plot":
            path += ".svg"
        else:
            path += ".json"

        return path


class IDPC(DPCD):
    """
    改进了度量方式、局部密度计算方式的 DPC 算法
    """

    def __init__(self, path, save_path="../../results/", use_cols=None, num=0, dc_method=1, dc_percent=1,
                 rho_method=1, delta_method=1, distance_method='irod', params=None, use_halo=False, plot=None):
        """
        初始化相关成员
        Parameters
        ----------
        path: 文件完整路径，csv 文件，如果是图片，也统一转化为 csv 储存像素点
        use_cols: 使用的列，两个属性(不含标签列)指定与否都可以；超过两列则需要指定读取的列(是否包含标签列)
        num: 聚类类簇数
        dc_method: 截断距离计算方法，默认为 1，即通过 dc_percent 指定
        dc_percent: 截断距离百分比数
        rho_method: 局部密度计算方法
        delta_method: 相对距离计算方法
        distance_method: 距离度量方式，默认为 irod
        params: 参数集合
        use_halo: 是否计算光晕点
        plot: 绘图句柄，仅有两个属性的数据集才可以做出数据原始结构图和聚类结果图
        """
        super(IDPC, self).__init__(path, save_path=save_path, use_cols=use_cols, num=num, dc_method=dc_method,
                                   dc_percent=dc_percent, rho_method=rho_method, delta_method=delta_method,
                                   distance_method=distance_method, params=params, use_halo=use_halo, plot=plot)

    def load_points_msg(self):
        """
        获取数据集相关信息，距离矩阵，ROD 距离列表，最小距离，最大距离
        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        return self.distance_rods()

    def rods_fun(self, i, j, rank_order_table, euclidean_table):
        """
        rod 及其改进算法
        Parameters
        ----------
        i: 第 i 个样本
        j: 第 j 个样本
        rank_order_table: 排序距离表
        euclidean_table: 欧式距离表

        Returns
        -------
        res: x1 与 x2 之间的距离
        """
        '''第 i 个样本'''
        x1 = rank_order_table[i, :]
        '''第 j 个样本'''
        x2 = rank_order_table[j, :]

        '''rod 及其改进算法需要的一些前置条件'''
        '''x1 样本的 id'''
        id_x1 = x1[0]
        '''x2 样本的 id'''
        id_x2 = x2[0]
        '''o_a_b，b 在 a 的排序列表中的索引位置'''
        o_a_b = numpy.where(x1 == id_x2)[0][0]
        '''o_b_a，a 在 b 的排序列表中的索引位置'''
        o_b_a = numpy.where(x2 == id_x1)[0][0]

        '''改进 rod'''
        l_a_b = (o_a_b + o_b_a) / (len(self.use_cols) - 1)
        '''高斯核'''
        k_a_b = math.exp(-(euclidean_table.at[i, j] / self.params["mu"]) ** 2)
        '''krod，改进 rod 与高斯核相结合'''
        res = l_a_b / k_a_b

        return res

    def get_rho(self, dc):
        """
        改进局部密度计算方式
        Parameters
        ----------
        dc: 截断距离

        Returns
        -------
        rho: 每个样本点的局部密度
        """
        '''深拷贝一份 self.dis_matrix 样本间距离矩阵'''
        matrix = self.dis_matrix.copy()
        '''按样本距离排序，这里按行排序'''
        matrix.sort()
        '''每个样本点的局部密度，k 近邻样本距离之和'''
        rho = int(self.params["k"]) / matrix[:, 1:int(self.params["k"]) + 1].sum(axis=1)

        return rho

    def get_file_path(self, dir_type):
        """
        保存文件路径
        Parameters
        ----------
        dir_type: 要保存的文件夹(data，plot，result)

        Returns
        -------
        path: 保存结果文件的路径
        """
        '''保存文件路径'''
        path = self.save_path + dir_type + "/" + self.data_name + "_" + self.distance_method + "_irho" + "/"

        '''判断文件夹是否存在'''
        if not os.path.isdir(path):
            '''创建文件夹'''
            os.mkdir(path)

        '''由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数'''
        '''删去 self.dc_method，因为在这个类里面，截断距离都是通过 self.dc_percent 指定的'''
        '''删去 self.rho_method，因为在这个类里面，局部密度计算方法将做出调整'''
        '''度量方法默认为 irod'''
        path += "dcp_" + str(self.dc_percent) + "__dem_" + str(self.delta_method) + "__ush_" + \
                str(int(self.use_halo)) + "__mu_" + str(self.params["mu"])

        '''根据不同的文件夹类型保存不同类型的文件'''
        if dir_type == "data":
            path += ".csv"
        elif dir_type == "plot":
            path += ".svg"
        else:
            path += ".json"

        return path


class DPCRHO(DPCD):
    """
    其他改进了局部密度计算方式的对比算法
    """

    def __init__(self, path, save_path="../../results/", use_cols=None, num=0, dc_method=1, dc_percent=1,
                 rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False, plot=None):
        """
        初始化相关成员
        Parameters
        ----------
        path: 文件完整路径，csv 文件，如果是图片，也统一转化为 csv 储存像素点
        use_cols: 使用的列，两个属性(不含标签列)指定与否都可以；超过两列则需要指定读取的列(是否包含标签列)
        num: 聚类类簇数
        dc_method: 截断距离计算方法
        dc_percent: 截断距离百分比数
        rho_method: 局部密度计算方法
        delta_method: 相对距离计算方法
        distance_method: 距离度量方式
        params: 参数集合
        use_halo: 是否计算光晕点
        plot: 绘图句柄，仅有两个属性的数据集才可以做出数据原始结构图和聚类结果图
        """
        super(DPCRHO, self).__init__(path, save_path=save_path, use_cols=use_cols, num=num, dc_method=dc_method,
                                     dc_percent=dc_percent, rho_method=rho_method, delta_method=delta_method,
                                     distance_method=distance_method, params=params, use_halo=use_halo, plot=plot)

    def load_points_msg(self):
        """
        获取数据集相关信息，距离矩阵，ROD 距离列表，最小距离，最大距离
        Returns
        -------
        dis_array: 样本间距离的矩阵
        min_dis: 样本间最小距离
        max_dis: 样本间最大距离
        """
        if self.distance_method in metric_way_set:
            return self.distance_standard()

    def get_rho(self, dc):
        """
        改进局部密度计算方式
        Parameters
        ----------
        dc: 截断距离

        Returns
        -------
        rho: 每个样本点的局部密度
        """
        '''每个样本点的局部密度'''
        rho = numpy.zeros(self.samples_num)

        if self.rho_method == 0:
            '''截断核'''
            for i in range(self.samples_num):
                '''到样本点 i 距离小于 dc 的点数量'''
                rho[i] = len(self.dis_matrix.loc[i, :][self.dis_matrix.loc[i, :] < dc]) - 1
        elif self.rho_method == 1:
            '''高斯核'''
            for i in range(self.samples_num):
                for j in range(self.samples_num):
                    if i != j:
                        rho[i] += math.exp(-(self.dis_matrix.at[i, j] / dc) ** 2)
        elif self.rho_method == 2:
            '''前 5% 的点'''
            for i in range(self.samples_num):
                '''排除异常值'''
                n = int(self.samples_num * 0.05)
                '''选择前 n 个离 i 最近的样本点'''
                rho[i] = math.exp(-(self.dis_matrix.loc[i].sort_values().values[:n].sum() / (n - 1)))
        elif self.rho_method == 3:
            '''DPC-KNN'''
            '''深拷贝一份 self.dis_matrix 样本间距离矩阵'''
            matrix = self.dis_matrix.copy()
            '''按样本距离排序，这里按行排序'''
            matrix.sort()
            '''k 近邻个样本，1 到 1 + self.params["mu"]'''
            exp = -(matrix[:, 1 + int(self.params["mu"])] ** 2).sum(axis=1) / int(self.params["mu"])
            '''局部密度'''
            for i in range(self.samples_num):
                rho[i] = math.exp(exp[i])

        return rho

    def get_file_path(self, dir_type):
        """
        保存文件路径
        Parameters
        ----------
        dir_type: 要保存的文件夹(data，plot，result)

        Returns
        -------
        path: 保存结果文件的路径
        """
        '''保存文件路径'''
        if self.rho_method in (0, 1, 2):
            '''DPC 标准方法'''
            path = self.save_path + dir_type + "/" + self.data_name + "_standard/"
        elif self.rho_method == 3:
            '''DPC-KNN'''
            path = self.save_path + dir_type + "/" + self.data_name + "_DPCKNN/"
        else:
            path = self.save_path + dir_type + "/" + self.data_name + "_" + self.distance_method + "/"

        '''判断文件夹是否存在'''
        if not os.path.isdir(path):
            '''创建文件夹'''
            os.mkdir(path)

        '''由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数'''
        '''删去 self.dc_method，因为在这个类里面，截断距离都是通过 self.dc_percent 指定的'''
        '''删去 self.rho_method，因为在这个类里面，局部密度方式体现在父文件夹名字里'''
        '''删去 self.distance_method，因为在这个类里面，度量方式是默认的'''
        path += "dcp_" + str(self.dc_percent) + "__dem_" + str(self.delta_method) + "__ush_" + str(int(self.use_halo))

        '''加上近邻数 k'''
        path += "__k_" + str(self.params["k"])

        '''根据不同的文件夹类型保存不同类型的文件'''
        if dir_type == "data":
            path += ".csv"
        elif dir_type == "plot":
            path += ".svg"
        else:
            path += ".json"

        return path
