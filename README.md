# Detection-of-particles-and-bubbles-in-a-three-phase-environment
检测盐水溶液中的微颗粒实时数据分析，解决工业应用中的痛点问题
1Original data 为OMIPA采集到的原始数据。
2Feature data 为matlab处理识别出的特征数据
3matlab信号去噪数据集设计源代码 为对原始数据进行小波去噪、形态学滤波、特征定义与提取、数据集构建 源码
4聚类源代码 为使用Python采用BIRCH聚类算法进行的特征聚类源码

本方案的特征定义清楚后，后续完全可以采用其他机器学习方案（监督、无监督）方案进行分类的研究。相对于聚类方案，依然有较为突出的标签，和有价值的应用场景。
