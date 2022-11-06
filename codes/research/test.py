import sys
import math
import scipy
import numpy
import pandas
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


class DPC(object):
    """
    经典 DPC 算法
    加载数据、计算矩阵矩阵
    计算截断距离、计算局部密度
    计算相对距离、确定距离中心
    样本分配、作图
    Args:
        object (_type_): _description_

    """

    def __init__(self, path, datas, num=0, dc_method=0, dc_percent=1, rho_method=1, delta_method=1, use_halo=False,
                 plot=None):
        """
        初始化函数
        Args:
            path (_type_): 文件路径
            data_name (_type_): 文件名
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_method (int, optional): 截断聚类计算方法. Defaults to 0.
            dc_percent (int, optional): 截断聚类百分比数. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 1.
            delta_method (int, optional): 相对距离计算方法. Defaults to 1.
            use_halo (_type_, optional): 光晕点计算. Defaults to false.
            plot (_type_, optional): 绘图句柄. Defaults to None.

        """
        self.path = path
        self.data_name = datas.split(".")[0]
        self.num = num
        self.dc_method = dc_method
        self.dc_percent = dc_percent
        self.rho_method = rho_method
        self.delta_method = delta_method
        self.use_halo = use_halo
        self.plot = plot
        ''''''
        self.border_b = list()

    def cluster(self):
        """
        运行算法
        """
        '''获取数据集的样本点，距离矩阵，欧式距离，最小距离，最大距离，样本数'''
        points, dis_matrix, dis_array, min_dis, max_dis, max_id = self.load_points_distance()
        '''计算截断距离 dc'''
        # print(points, dis_matrix, dis_array, min_dis, max_dis, max_id)
        dc = self.get_dc(dis_matrix, dis_array, min_dis, max_dis, max_id)
        # print(dc)
        '''计算局部密度 rho'''
        rho = self.get_rho(dis_matrix, max_id, dc)
        # print(rho)
        '''计算相对距离 delta'''
        delta = self.get_delta(dis_matrix, max_id, rho)
        # print(delta)
        '''确定聚类中心和 gamma(局部密度与相对距离的乘积)'''
        center, gamma = self.get_center(dis_matrix, rho, delta, dc, max_id)
        # print(center)
        # print(gamma)
        '''非聚类中心样本点分配'''
        cluster = self.assign(dis_matrix, dc, rho, delta, center, max_id)
        # print(cluster)
        halo = list()
        if self.use_halo:
            cluster, halo = self.get_halo(
                dis_matrix, rho, cluster, center, dc, max_id)

        '''绘图'''
        if self.plot is None:
            '''单一数据集绘图'''
            fig, axes = plt.subplots(2, 2, figsize=(18.6, 18.6))
            fig.subplots_adjust(left=0.05, right=0.95)
            axes[0][1].set_title('dc-' + str(self.dc_method) + '(' + str(dc) + ')' ' | rho-' +
                                 str(self.rho_method) + ' | delta-' + str(self.delta_method))
            '''第一张图'''
            self.draw_points(points, [], axes[0][0])
            '''第二张图'''
            self.draw_rho_delta(rho, delta, center, axes[0][1])
            '''第三张图'''
            self.draw_gamma(rho, delta, axes[1][0])
            '''第四张图'''
            self.draw_cluster(cluster, halo, points, axes[1][1])
            plt.show()
        else:
            '''全部数据集绘图'''
            self.draw_cluster(cluster, halo, points, None)

    def distance(self, points, metric):
        """
        样本度量方法
        Args:
            points (_type_): 样本点
            metric (_type_): 距离度量方法

        Returns:
            dis_array: 距离度量矩阵(列表)
        """
        dis_array = sch.distance.pdist(points, 'euclidean')
        return dis_array

    def load_points_distance(self):
        """
        获取数据集的样本点，距离矩阵，欧式距离，最小距离，最大距离，样本数
        Returns:
            points : 样本点
            dis_matrix : 距离矩阵
            dis_array : 欧式距离
            min_dis : 最小距离
            max_dis : 最大距离
            max_id : 最大点数

        """
        points = pandas.read_csv(self.path, sep="\t", usecols=[0, 1])
        # 设置为成类成员，样本个数
        max_id = len(points)

        dis_matrix = pandas.DataFrame(numpy.zeros((max_id, max_id)))
        '''维度为 max_id * (max_id) / 2'''
        dis_array = self.distance(points, 'euclidean')

        num = 0
        for i in range(max_id):
            for j in range(i + 1, max_id):
                dis_matrix.at[i, j] = dis_array[num]
                dis_matrix.at[j, i] = dis_matrix.at[i, j]
                num += 1
        '''最小距离'''
        min_dis = dis_matrix.min().min()
        '''最大距离'''
        max_dis = dis_matrix.max().max()

        return points, dis_matrix, dis_array, min_dis, max_dis, max_id

    def get_dc(self, dis_matrix, dis_array, min_dis, max_dis, max_id) -> float:
        """
        计算截断距离
        Args:
            dis_matrix : 距离矩阵
            dis_array : 上三角距离矩阵
            min_dis : 最小距离
            max_dis : 最大距离
            max_id : 点数

        Returns:
            float: 截断距离
        """
        lower = self.dc_percent / 100
        upper = (self.dc_percent + 1) / 100
        if self.dc_method == 0:
            while 1:
                dc = (min_dis + max_dis) / 2
                '''上三角矩阵'''
                neighbors_percent = len(
                    dis_array[dis_array < dc]) / (((max_id - 1) ** 2) / 2)
                if neighbors_percent >= lower and neighbors_percent <= upper:
                    return dc
                elif neighbors_percent > upper:
                    max_dis = dc
                elif neighbors_percent < lower:
                    min_dis = dc

    def get_rho(self, dis_matrix, max_id, dc) -> numpy.array:
        """
        计算局部密度
        Args:
            dis_matrix (_type_): 距离矩阵
            max_id (_type_): 点数
            dc (_type_): 截断距离

        Returns:
            numpy.array: 局部密度
        """
        rho = numpy.zeros(max_id)

        for i in range(max_id):
            if self.rho_method == 0:
                '''和点 i 距离小于 dc 的点的数量'''
                rho[i] = len(dis_matrix.loc[i, :][dis_matrix.loc[i, :] < dc]) - 1
            elif self.rho_method == 1:
                '''高斯核'''
                for j in range(max_id):
                    if i != j:
                        rho[i] += math.exp(-(dis_matrix.at[i, j] / dc) ** 2)
            elif self.rho_method == 2:
                '''排除异常值'''
                n = int(max_id * 0.05)
                '''选择前 n 个离 i 最近的样本点'''
                rho[i] = math.exp(-(dis_matrix.loc[i].sort_values().values[:n].sum() / (n - 1)))

        return rho

    def get_delta(self, dis_matrix, max_id, rho) -> numpy.array:
        """
        计算相对距离
        Args:
            dis_matrix (_type_): 距离矩阵
            max_id (_type_): 点数
            rho (_type_): 局部密度

        Returns:
            delta: 相对距离
        """
        delta = numpy.zeros(max_id)

        if self.delta_method == 0:
            '''不考虑 rho 相同且同为最大'''
            for i in range(max_id):
                rho_i = rho[i]
                '''rho 大于 rho_i 的点'''
                j_list = numpy.where(rho > rho_i)[0]
                if len(j_list) == 0:
                    '''局部密度最大的点'''
                    delta[i] = dis_matrix.loc[i, :].max()
                else:
                    '''局部密度大于 i 且离 i 最近点的索引'''
                    min_dis_idx = dis_matrix.loc[i, j_list].idxmin()
                    delta[i] = dis_matrix.at[i, min_dis_idx]
        elif self.delta_method == 1:
            '''考虑 rho 相同且同为最大'''
            '''局部密度从小到大排序，并且反转，返回对应值的索引'''
            rho_order_idx = rho.argsort()[-1::-1]
            for i in range(1, max_id):
                '''对应 rho 的索引'''
                rho_idx = rho_order_idx[i]
                '''j < i 的排序索引值'''
                j_list = rho_order_idx[:i]
                '''返回第 rho_idx 行，j_list 列中最小值对应的索引'''
                min_dis_idx = dis_matrix.loc[rho_idx, j_list].idxmin()
                '''比 i 大，且离 i 最近的距离即为 i 的相对距离'''
                delta[rho_idx] = dis_matrix.at[rho_idx, min_dis_idx]
            '''相对距离最大的点'''
            delta[rho_order_idx[0]] = delta.max()

        return delta

    def get_center(self, dis_matrix, rho, delta, dc, max_id):
        """
        获取距离中心和 gamma
        Args:
            dis_matrix (_type_): 距离矩阵
            rho (_type_): 局部密度
            delta (_type_): 相对距离
            dc (_type_): 截断距离
            max_id (_type_): 点数

        Returns:
            center: 聚类中心列表
            gamma: rho * delta
        """
        gamma = rho * delta
        gamma = pandas.DataFrame(gamma, columns=['gamma']).sort_values(
            'gamma', ascending=False)
        '''取 gamma 最大的前 self.num 个点作为聚类中心'''
        center = numpy.array(gamma.index)[:self.num]
        '''另一种方法'''
        # center = gamma[gamma.gamma>threshold].loc[:,'gamma'].index
        return center, gamma

    def assign(self, dis_matrix, dc, rho, delta, center, max_id):
        """
        非距离中心样本点分配
        Args:
            dis_matrix (_type_): 距离矩阵
            dc (_type_): 截断距离
            rho (_type_): 局部密度
            delta (_type_): 相对距离
            center (_type_): 距离中心点
            max_id (_type_): 样本数

        Returns:
            cluster: dict(center, points)
        """
        cluster = dict()
        for c in center:
            cluster[c] = list()

        link = dict()
        '''局部密度排序'''
        order_rho_idx = rho.argsort()[-1::-1]
        for i, v in enumerate(order_rho_idx):
            if v in center:
                '''聚类中心'''
                link[v] = v
                continue
            '''非聚类中心的点'''
            '''前 i 个局部密度排序索引值，也就是局部密度大于 rho[v] 的索引列表'''
            rho_idx = order_rho_idx[:i]
            '''在局部密度大于 rho[v] 的点中，距离从小到大排序的第一个索引，也是离得最近的点(不一定是聚类中心)'''
            link[v] = dis_matrix.loc[v, rho_idx].sort_values().index.tolist()[
                0]

        '''分配非聚类中心的点'''
        for k, v in link.items():
            '''使用 c 记录 k 离得最近的点 v'''
            c = v
            '''c 不是聚类中心'''
            while c not in center:
                '''c 更新为 c 离得最近的点 link[c]，一步步迭代，直到找到 c 对应的聚类中心'''
                c = link[c]
            '''c 是聚类中心，分配当前点 k 到 c 中'''
            cluster[c].append(k)

        # 最近中心分配
        # for i in range(self.num):
        #     c = dis_matrix.loc[i, center].idxmin()
        #     cluster[c].append(i)

        return cluster

    def get_halo(self, dis_matrix, rho, cluster, center, dc, max_id):
        """
        获取光晕点
        Args:
            dis_matrix (_type_): 距离矩阵
            rho (_type_): 局部密度
            cluster (_type_): 聚类结果
            center (_type_): 聚类中心
            dc (_type_): 截断距离
            max_id (_type_): 样本数

        Returns:
            cluster: 聚类结果
            halo: 光晕点
        """
        '''所有点'''
        all_points = set(list(range(max_id)))
        for c, points in cluster.items():
            '''属于其他聚类的点'''
            others_points = list(set(all_points) - set(points))
            border = list()
            for p in points:
                '''到其他聚类点的距离小于 dc'''
                if dis_matrix.loc[p, others_points].min() < dc:
                    border.append(p)
            if len(border) != 0:
                '''边界域中密度最大的值'''
                # rbo_b = border[rho[border].argmax()]
                '''边界域中密度最大的点'''
                point_b = border[rho[border].argmax()]
                self.border_b.append(point_b)
                '''边界域最大密度'''
                rho_b = rho[point_b]
                '''筛选可靠性高的点'''
                filter_points = numpy.where(rho >= rho_b)[0]
                '''该聚类中可靠性高的点'''
                points = list(set(filter_points) & set(points))
                cluster[c] = points

        '''halo'''
        cluster_points = set()
        for c, points in cluster.items():
            cluster_points = cluster_points | set(points)
        '''光晕点'''
        halo = list(set(all_points) - cluster_points)

        return cluster, halo

    def draw_points(self, points, center, plot):
        """
        绘图：原始数据分布图
        Args:
            points (_type_): 样本点
            center (_type_, optional): 聚类中心
            plot (_type_): 绘图句柄

        """
        plot.scatter(points.loc[:, 'x'],
                     points.loc[:, 'y'], c='k')
        if len(center) != 0:
            center_p = points.loc[center, :]
            plot.scatter(center_p.loc[:, 'x'],
                         center_p.loc[:, 'y'], c='r', s=numpy.pi * 8 ** 2)
        plot.set_title("raw data")

    def draw_rho_delta(self, rho, delta, center, plot):
        """
        绘图：局部密度与相对距离
        Args:
            rho (_type_): 局部密度
            delta (_type_): 相对距离
            center (_type_): 聚类中心
            plot (_type_): 绘图句柄

        """
        plot.scatter(rho, delta, label='rho-delta', c='k', s=5)
        plot.set_xlabel('rho')
        plot.set_ylabel('delta')
        '''聚类中心点的局部密度'''
        center_rho = rho[center]
        '''聚类中心点的相对距离'''
        center_delta = delta[center]
        '''随机数种子，设置颜色'''
        colors = numpy.random.rand(len(center), 3)
        plot.scatter(center_rho, center_delta, c=colors)
        plot.legend()

    def draw_gamma(self, rho, delta, plot):
        """
        绘图：局部密度与相对距离的乘积
        Args:
            rho (_type_): 局部密度
            delta (_type_): 相对距离
            plot (_type_): 绘图句柄

        """
        gamma = pandas.DataFrame(
            rho * delta, columns=['gamma']).sort_values('gamma', ascending=False)
        plot.scatter(range(len(gamma)),
                     gamma.loc[:, 'gamma'], label='gamma', s=5)
        # plot.hlines(avg, 0, len(gamma), 'b', 'dashed')
        plot.set_xlabel('num')
        plot.set_ylabel('gamma')
        plot.set_title('gamma')
        plot.legend()

    def draw_cluster(self, cluster, halo, points, plot):
        """
        绘制绘图结果
        Args:
            cluster (_type_): 聚类结果
            halo (_type_): 光晕点
            points (_type_): 样本点
            plot (_type_): 绘图句柄

        """
        cluster_points = dict()
        colors = dict()
        numpy.random.seed(10)

        for k, v in cluster.items():
            '''同一类中的样本点'''
            cluster_points[k] = points.loc[cluster[k], :]
            colors[k] = numpy.random.rand(3)


        for k, v in cluster_points.items():
            plot.scatter(v.loc[:, 'x'], v.loc[:, 'y'], c=colors[k], alpha=0.5)
            plot.scatter(v.at[k, 'x'], v.at[k, 'y'],
                         c=colors[k], s=numpy.pi * 8 ** 2)

        if len(halo) != 0:
            noise_pointer = points.loc[halo, :]
            plot.scatter(noise_pointer.loc[:, 'x'],
                         noise_pointer.loc[:, 'y'], c='k')
            border_b = points.loc[self.border_b, :]
            plot.scatter(
                border_b.loc[:, 'x'], border_b.loc[:, 'y'], c='k', s=numpy.pi * 4 ** 2)
        plot.set_title(self.data_name)


if __name__ == "__main__":
    """"""
    p = "../../dataSet/test.dat"
    d = DPC(p, "test", num=3)
    d.cluster()
