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
        * [demo](dataSet/data/demo)
        * [mnist](dataSet/data/mnist)
        * [uci](dataSet/data/uci)
    * [experiment](dataSet/experiment/) 实验需要用的数据，从 [data](dataSet/data/) 中提取，加入了相关参数
        * [demo](dataSet/data/demo)
        * [mnist](dataSet/data/mnist)
        * [uci](dataSet/data/uci)
* [notes](notes) 笔记块
* [results](results) 结果块，均以 pandas.DataFrame 形式存放，以追加的形式增加到现有数据中
    * [demo](results/demo) 结构同下
    * [mnist](results/mnist) 结构同下
    * [uci](results/uci) 存放结果(多进程下，每一个类创建属于自己的结果文件)
        * [data](results/demo/data) 存放聚类结果数据，只存放预测的标签列(列名为此次算法运行的参数，列内容为标签，顺序与原始数据索引一致)
        * [plot](results/demo/plot) 存放聚类结果图
        * [result](results/demo/result) 存放聚类结果指标
        * [analyze](results/demo/analyze) 存放分析结果(列名为参数+聚类指标，行为每一次不同参数下的实验结果)
* [README](README.md) readme

---
