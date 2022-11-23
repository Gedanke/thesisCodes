# result

---

## build

euclidean、rod、krod、irod

blobs:
四个算法一样，都是 1
同上，这个数据集没有参考价值

---

param = {
    "norm": 0,
    "mu": 1,
    "sigma": 0.7,
    "make": {
        "samples": 400,
        "features": 8,
        "classes": 5,
        "noise": 0.15,
        "random": 4
    }
}

circles:
结果差不多
同上
噪声级别不要太高
mu=10

此时 irod>rod>krod==e

gaussian_quantiles：
后三者差不多，比欧式距离高一点，但结果都很低
同上
mu=1
有时候会反过来

hastie_10_2：
后三者差不多，e>irod>=krod>rod，比欧式距离低一点
同上
mu=1
有时候会反过来
irod > krod=e > rod


moons：noise=0.2
欧式最低，rod 高不少，krod 比 rod 高，irod 比 krod 高
0.60、0.68、0.74、0.77
有时候结果会反过来，这种情况偏少一点

irod 比 rod、krod 要好一些，几个百分点的提升
也有负提升