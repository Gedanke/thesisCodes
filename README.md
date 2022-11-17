# readme

学位论文的相关代码

---

## 代码结构

* [codes](codes) 代码块
    * [others](codes/others) 其他对比算法
    * [research](codes/research) dpc 基类、dpc 改进的派生类、数据预处理、结果分析等代码
* [dataSet](dataSet) 数据块
    * [raw](dataSet/raw) 原始数据
        * [demo](dataSet/raw/demo)
        * [mnist](dataSet/raw/mnist)
        * [uci](dataSet/raw/uci)
    * [data](dataSet/data) 预处理完成后的数据
        * [build](dataSet/data/build)
        * [demo](dataSet/data/demo)
        * [mnist](dataSet/data/mnist)
        * [uci](dataSet/data/uci)
    * [experiment](dataSet/experiment/) 实验需要用的数据，从 [data](dataSet/data/) 中提取，加入了相关参数
        * [build](dataSet/data/build)
        * [demo](dataSet/data/demo)
        * [mnist](dataSet/data/mnist)
        * [uci](dataSet/data/uci)
* [notes](notes) 笔记块
* [results](results) 结果块
    * [data](results/data) 存放聚类结果数据
    * [plot](results/plot) 存放聚类结果图
    * [result](results/result) 存放聚类结果指标
    * [analyze](results/analyze) 存放分析结果
* [README](README.md) readme

---
