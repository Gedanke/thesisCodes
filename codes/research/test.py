# -*- coding: utf-8 -*-

from dpc import *

PATH = "../../dataSet/"
DATAS = ['test.dat']

metric_way_set = {
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming",
    "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao",
    "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"
}


def distance_(samples: pandas.DataFrame, metric_way='euclidean') -> numpy.ndarray:
    """
    样本度量方法
    Parameters
    ----------
    samples: pandas.DataFrame 只含属性列
    metric_way: 属性度量方法

    Returns
    -------
    dis_array: 返回的依然是距离度量的矩阵(列表)
    维度为 samples_num * (samples_num - 1) / 2
    """
    # '''此处使用的是欧式距离'''
    # dis_array = sch.distance.pdist(samples, metric_way)
    # return dis_array

    '''样本个数'''
    samples_num = len(samples)
    '''预分配空间'''
    dis_array = numpy.zeros((int(samples_num * (samples_num - 1) / 2)))

    idx = 0
    for i in range(0, samples_num):
        for j in range(i + 1, samples_num):
            dis_array[idx] = metric_fun(samples.iloc[i, :], samples.iloc[j, :])
            idx += 1

    return dis_array


def metric_fun(x1, x2, metric_way='rod'):
    """

    Parameters
    ----------
    x1:
    x2:
    metric_way:
    Returns
    -------

    """
    res = -1
    mu = 1
    '''判断度量方法'''
    if metric_way == 'rod':
        idx1 = x1[0]
        idx2 = x2[0]
        '''o_a_b，b 在 a 的排序列表中的索引位置'''
        o_a_b = numpy.where(x1 == idx2)[0][0]
        '''o_b_a，a 在 b 的排序列表中的索引位置'''
        o_b_a = numpy.where(x2 == idx1)[0][0]
        '''d_a_b'''
        '''先切片，是 a 的排序列表中 [a,b] 的索引'''
        slice_a_b = x1[0:o_a_b + 1]
        '''索引在 b 中的位置的序数和'''
        d_a_b = sum(numpy.where(x2 == slice_a_b[:, None])[-1])
        '''d_b_a'''
        '''先切片，是 b 的排序列表中 [b,a] 的索引'''
        slice_b_a = x2[0:o_b_a + 1]
        '''索引在 a 中的位置的序数和'''
        d_b_a = sum(numpy.where(x1 == slice_b_a[:, None])[-1])
        res = (d_a_b + d_b_a) / min(o_a_b, o_b_a)
    elif metric_way == "krod":
        idx1 = x1[0]
        idx2 = x2[0]
        '''o_a_b，b 在 a 的排序列表中的索引位置'''
        o_a_b = numpy.where(x1 == idx2)[0][0]
        '''o_b_a，a 在 b 的排序列表中的索引位置'''
        o_b_a = numpy.where(x2 == idx1)[0][0]
        l_a_b = o_a_b + o_b_a
        k_a_b = math.exp(-((x1 - x2) / mu) ** 2)
        res = l_a_b * (1 / k_a_b)
    elif metric_way == "irod":
        idx1 = x1[0]
        idx2 = x2[0]
        '''o_a_b，b 在 a 的排序列表中的索引位置'''
        o_a_b = numpy.where(x1 == idx2)[0][0]
        '''o_b_a，a 在 b 的排序列表中的索引位置'''
        o_b_a = numpy.where(x2 == idx1)[0][0]

        #
    return res


def distance(path):
    """

    Parameters
    ----------
    path

    Returns
    -------

    """
    samples = pandas.read_csv(path, usecols=[0, 1])
    '''样本个数'''
    samples_num = len(samples)
    '''距离矩阵初始化'''
    dis_matrix = pandas.DataFrame(numpy.zeros((samples_num, samples_num)))
    '''预分配空间'''
    dis_array = sch.distance.pdist(samples, 'euclidean')

    '''对距离矩阵的处理'''
    num = 0
    for i in range(samples_num):
        for j in range(i + 1, samples_num):
            '''赋值'''
            dis_matrix.at[i, j] = dis_array[num]
            '''处理对角元素'''
            dis_matrix.at[j, i] = dis_matrix.at[i, j]
            num += 1

    rank_order_table = numpy.array(dis_matrix).argsort()
    """
    [[ 0  1  2  3  7  6  5  4  8  9 10 11]
 [ 1  2  0  3  7  6  5  4  8  9 10 11]
 [ 2  1  3  0  7  6  8  5  9  4 10 11]
 [ 3  2  1  0  7  8  6  9 10  5 11  4]
 [ 4  5  6  7 11 10  9  0  8  1  2  3]
 [ 5  4  6  7 11  0 10  1  9  2  3  8]
 [ 6  7  5  4  0  1  2 11  3 10  9  8]
 [ 7  6  5  4  0  1  2  3 11 10  9  8]
 [ 8  9 10 11  3  2  4  1  5  6  0  7]
 [ 9 10  8 11  4  3  5  6  2  7  1  0]
 [10  9 11  8  4  5  6  3  7  2  1  0]
 [11 10  9  8  4  5  6  7  3  2  1  0]]
    """
    print(metric_fun(rank_order_table[4, :], rank_order_table[9, :], "rod"))
    # l = numpy.array([0, 3, 2, 1, 4])
    # search = numpy.array([3, 4])
    # print(numpy.where(l == search[:, None])[-1])
    # print(l[0:3])
    # print(l)
    print(dis_array.shape)


def test():
    """

    Returns
    -------

    """
    p = PATH + "data/test.csv"
    distance(p)
    # d = DPC(p, num=3, use_halo=True)
    # d.cluster()
    # print(d.dis_matrix)




if __name__ == "__main__":
    """"""