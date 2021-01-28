#lunwen 后半部分的图
#无监督学习 聚类的实现，进行 2类点 的分类 Birch 主要实现
# 合并混合的颗粒与对应气泡的聚类区分。剔除气泡，得到粒径统计分布图。
#无标签的分类 。分类结果 与 标签 和 贝克曼结果   都做了对比
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.cluster import Birch
from sklearn import metrics
from pylab import *
import xlrd

import matplotlib as mpl
from matplotlib import colors

def huitu_histograms(file):
    file_title = file.columns.values.tolist()  # 获得列名
    n_bins =60
    for j in range(0,3):
        fig, axs = plt.subplots(1, 6, sharey=False, tight_layout=True, figsize=(16, 8))
        for i in range(0+6*j, 6+6*j):
            axs[i-6*j].hist(file[file_title[i]], bins=n_bins, color='deepskyblue')
            axs[i-6*j].set_title(file_title[i], fontsize=15, color='royalblue')
    plt.show()

def Standardized_data(file):
    # Sd_file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 方法一
    Sd_file = file.apply(lambda x: (x - np.mean(x)) / (np.std(x)))  # 方法二
    return Sd_file

def Clustering_deletion(Sd_file,file,num):

    print(len(Sd_file))

    x = Sd_file.values[:, [3,15,8,2,17]].astype('float')  # 最完美的特征组合 3,15,8,14
    # ********************kmeans聚类*****************************
    # 创建kmeans聚类
    kmeans = Birch(n_clusters=2, threshold = 0.05)
    y_kmeans = kmeans.fit_predict(x)

    # 用Calinski-Harabasz Index评估的聚类分数:
    print(metrics.calinski_harabasz_score(x, y_kmeans))

    # 画图一
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['b', 'g', 'r'])
    fig=figure(num)
    border_width = 0.2
    ax_size = [0.25, 0.15,
               0.5, 0.7]
    ax  = fig.add_axes(ax_size)
    q = 0  # 记录选择的  特征向量
    p = 1
    fontsize_title = 40
    fontsize_axis = 40
    tick_size = 20
    legend_size = 25

    ax.scatter(x[y_kmeans == 0, q], x[y_kmeans == 0, p], s=100, c='red', label='Prediction of Bubblet')
    ax.scatter(x[y_kmeans == 1, q], x[y_kmeans == 1, p], s=100, c='blue', label='Prediction of Particle')

    x_train_feature ='Feature 1', 'Feature 2','Feature3_I16_T_skewness','I19_EPd3','I20_EPd4'
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Controids')
    plt.legend(loc='upper left',prop={'family': 'Times New Roman', 'size': legend_size})
    plt.tick_params(labelsize=tick_size)  # 坐标轴刻度数字大小
    plt.xlabel(x_train_feature[0],fontproperties = 'Times New Roman', fontsize=fontsize_axis)
    plt.ylabel(x_train_feature[1],fontproperties = 'Times New Roman', fontsize=fontsize_axis)
    plt.title('BIRCH used in data clustering',fontproperties = 'Times New Roman', fontsize=fontsize_title)

    #######################################################################
    # 画直方图  查看聚类分类后的结果
    x_real, y_real = np.split(file, (21,), axis=1)  # 读取为归一化的幅值数据

    # the histogram of the data
    sqort_3 = []  # 原始真实混杂的颗粒分布
    for i in range(0, len(Sd_file.values[:, q])):
        sqort_3.append(math.pow(x_real.values[i, q], 1 / 3) * 20.894 - 0.35)  # 找出对应关系

    x_kmeans0_sqort3 = []  # 记录k-means聚类为0的点的索引
    x_kmeans1_sqort3 = []
    kmeans_index_0 = np.where(y_kmeans == 0)[0]
    kmeans_index_1 = np.where(y_kmeans == 1)[0]

    for i in kmeans_index_0:
        x_kmeans0_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 - 0.35)

    for i in kmeans_index_1:
        x_kmeans1_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 - 0.35)
    plt.show()
    return x_kmeans0_sqort3,x_kmeans1_sqort3

def kmeans_hist(file_kmeans,x,y):

    plt.hist(file_kmeans, bins=90, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             # log=False, color="#FF6600", label="On-line ( Identified Bubble )", stacked=False)
             # log = False, color = "#FF6600", label = "On-line ( Particle + Removing Bubble )", stacked = False)
             log = False, color = "#FF6600", label = "On-line ( Particle + Bubble )", stacked = False)

    plt.bar(x, y, width=0.5, label="Off-line ( Particle )", color="#3366FF", alpha=0.85)
    font_size = 40  # 显示字体和坐标数字大小
    plt.legend(prop={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel("Diameter(μm)", fontproperties = 'Times New Roman', fontsize=font_size+5)
    plt.ylabel("Number", fontproperties = 'Times New Roman', fontsize=font_size+5)
    plt.xticks(fontsize=font_size-5)
    plt.yticks(fontsize=font_size-5)
    plt.show()

#******************************读取文件，划分数据集************************
## 读取原始-未归一化数据
if __name__ == "__main__":

    data10 = pd.read_excel("D:/20thesis/826/data_813/Classification - No validation/813_200_60-80_qipao.xlsx", encoding='gdk') #80-qipao

    index_80 = data10.loc[data10['I1_mv']>35].index #存放索引
    index_60 = data10.loc[data10['I1_mv']<35].index #存放索引

    file1=data10.iloc[index_60] #通过 iloc获得dataframe,再合并前面的特征 和后面的标签
    file2=data10.iloc[index_80]

    Sd_file1=Standardized_data(file1) #标准化
    Sd_file2=Standardized_data(file2) #标准化

    file1_kmeans0_D, file1_kmeans1_D = Clustering_deletion(Sd_file1,file1,1) #将归一化的数据，进行聚类分析，找到颗粒的索引，绘制
                                        #粒径分布图
    file2_kmeans0_D, file2_kmeans1_D = Clustering_deletion(Sd_file2,file2,2)

    file_kmeans0_D = file1_kmeans0_D + file2_kmeans0_D

    file_kmeans1_D = file1_kmeans1_D + file2_kmeans1_D

    file_kmeans2_D = file1_kmeans0_D + file2_kmeans1_D

    file_kmeans3_D = file1_kmeans1_D + file2_kmeans0_D

    file_kmeans_D = file1_kmeans0_D + file2_kmeans0_D + file1_kmeans1_D + file2_kmeans1_D

    ##  绘图 贝克曼的图 归一化  #########################################
    xx3 = xlrd.open_workbook(r"D:\20thesis\826\data_813\beckman_200_60-80_2.xls")
    sheet1 = xx3.sheet_by_name("Sheet1")
    x3 = sheet1.col_values(0, 0, 86)
    y3 = sheet1.col_values(1, 0, 86)

    def Standardized_data(file):
        x = file
        Sd_file = np.array(x) / 5#sum(x)  # 方法一
        return Sd_file

    y4 = Standardized_data(y3)

    #观察一下 在线数据 kmeans聚类的四个分部 和 贝克曼的对比结果。
    plt.figure(10)
    kmeans_hist(file_kmeans0_D, x3, y4)
    plt.figure(11)
    kmeans_hist(file_kmeans1_D, x3, y4)
    plt.figure(12)
    kmeans_hist(file_kmeans2_D, x3, y4)
    plt.figure(13)
    kmeans_hist(file_kmeans3_D, x3, y4)
    plt.figure(14)
    kmeans_hist(file_kmeans_D, x3, y4)  # 总体对比 原始在线 和 贝克曼 离线的

    plt.show()





