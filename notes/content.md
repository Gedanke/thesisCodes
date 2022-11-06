# content

* 局部密度优化
* 优化聚类中心
* 样本分配策略优化

评价指标分为外部指标和内部指标两种，外部指标指评价过程中需要借助数据真实情况进行对比分析，内部指标指不需要其他数据就可进行评估的指标

![](%E8%81%9A%E7%B1%BB%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87.jpg)

参考链接

* [https://blog.csdn.net/weixin_45317919/article/details/121472851](https://blog.csdn.net/weixin_45317919/article/details/121472851)
* [https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/__init__.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/__init__.py)
* [https://blog.csdn.net/qq_27825451/article/details/94436488](https://blog.csdn.net/qq_27825451/article/details/94436488)
* [https://blog.csdn.net/howhigh/article/details/73928635](https://blog.csdn.net/howhigh/article/details/73928635)

---

## 关于代码

位于 [dpc.py](../codes/dpc.py) 文件中的 DPC 类

该类不需要对数据进行预处理，输入的是处理好的数据，高内聚，低耦合，对于算法的改动只需要改动或者实现若干函数即可

这里采用继承的方式实现不同的 DPC 算法，继承 DPC 算法基类来实现不同的改进

该类能够完成对数据的聚类，保存聚类结果，作图，包括原始数据分布图，决策图，gamma 图，结果图等等，还能得到所有的聚类结果指标





---
