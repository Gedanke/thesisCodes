# -*- coding: utf-8 -*-

from dpc import *

rod_method_set = {
    "rod", "krod", "irod"
}


class IDPC(DPC):
    """
    改进了度量方式的 DPC 算法
    主要改进点集中在 self.load_points_msg 函数内
    并增加了一些其他度量方式进行对比、其他方法基本上保存一致
    """

    def __init__(self, path, use_cols=None, num=0, dc_method=0, dc_percent=1, rho_method=1, delta_method=1,
                 distance_method='euclidean', params=None, use_halo=False, plot=None):
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
        use_halo: 是否计算光晕点
        plot: 绘图句柄，仅有两个属性的数据集才可以做出数据原始结构图和聚类结果图
        """
        '''使用父类的构造方法'''
        super(IDPC, self).__init__(path, use_cols, num, dc_method, dc_percent, rho_method, delta_method, use_halo, plot)

        '''其他参数'''
        if params is None:
            params = {}
        self.params = params
        '''距离度量方式'''
        self.distance_method = distance_method

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
        """
        在 euclidean_table 中，每 i 行都是第 i 个元素到其他元素间的欧式距离
        使用 argsort() 索引降序后
        在 rank_order_table 内，第 i 行便是第 i 个元素到其他元素(包括了自己)的距离升序排列并对应元素在 euclidean_table 内的索引
        因此，在 rank_order_table 第 i 行内，第 0 个元素是离第 i 个元素最近的元素索引
        便是第 i 个样本本身，因为自己到自己的距离是最小的
        以此类推，第 i 行的第 1 个元素是离第 i 个元素第二近的元素索引......直到最后一个元素是离得最远得
        需要注意得是，rank_order_table 内储存的是元素升序的索引，不是对应的欧式距离
        如果想知道第 j 近的邻居到当前样本的距离，需要先在 rank_order_table 中拿到第 j 个位置上的索引
        通过索引去访问 euclidean_table，才能得到对应的距离
        总结而言，rank_order_table[i][j] 离表示第 i 个样本第 j 近的样本索引位置或者说就是第 rank_order_table[i][j] 个样本
        离表示第 i 个样本第 j 近的样本的距离为 euclidean_table[i][rank_order_table[i][j]]
        """
        rank_order_table = numpy.array(euclidean_table).argsort()

        '''对距离矩阵进行处理'''
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                '''用新的度量方式得到的样本间距离覆盖掉 dis_array'''
                dis_array[num] = self.rods_fun(rank_order_table[i, :], rank_order_table[j, :])
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

    def rods_fun(self, x1, x2):
        """
        rod 及其改进算法
        Parameters
        ----------
        x1: 样本 1
        x2: 样本 2

        Returns
        -------
        res: x1 与 x2 之间的距离
        """
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
            k_a_b = math.exp(-sum(((x1 - x2) / self.params["mu"]) ** 2))
            '''krod，改进 rod 与高斯核相结合'''
            res = l_a_b / k_a_b
        elif self.distance_method == "irod":
            '''先切片，a 的排序列表中 (a, b) 的索引列表'''
            slice_a_b = x1[0:o_a_b + 1]
            '''先切片，b 的排序列表中 [b, a] 的索引列表'''
            slice_b_a = x2[0:o_b_a + 1]
            '''求交集'''
            intersection = numpy.intersect1d(slice_a_b, slice_b_a, assume_unique=True, return_indices=True)
            '''结果'''
            res = (sum(intersection[1]) + sum(intersection[2])) / math.exp(-sum(((x1 - x2) / self.params["mu"]) ** 2))

        return res
