#lunwen 前半部分的图
#无监督学习 聚类的实现，进行 2类点 的分类 Birch 主要实现
# 合并混合的颗粒与对应气泡的聚类区分。剔除气泡，得到粒径统计分布图。
#有标签的分类 。分类结果 与 标签 和 贝克曼结果   都做了对比
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn import metrics
from pylab import *
import xlrd
import time

import matplotlib as mpl
from matplotlib import colors
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def huitu_histograms(file):
    file_title = file.columns.values.tolist()  # 获得列名
    n_bins =60

    for j in range(0,3):
        fig, axs = plt.subplots(1, 6, sharey=False, tight_layout=True, figsize=(16, 8))
        for i in range(0+6*j, 6+6*j):
            axs[i-6*j].hist(file[file_title[i]], bins=n_bins, color='deepskyblue') #deepskyblue
            axs[i-6*j].set_title(file_title[i], fontsize=15, color='royalblue')

    fig, axs = plt.subplots(1, 6, sharey=False, tight_layout=True, figsize=(16, 8))
    for i in range(0,3):
        axs[i].hist(file[file_title[18+i]], bins=n_bins, color='orangered')
        axs[i].set_title(file_title[18+i], fontsize=15, color='royalblue')

    plt.show()

def Standardized_data(file):
    # Sd_file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 方法一
    Sd_file = file.apply(lambda x: (x - np.mean(x)) / (np.std(x)))  # 方法二
    return Sd_file

def Clustering_deletion(Sd_file,file,num):
    x1, y1 = np.split(Sd_file,  # 要切分的数组
                      (21,),  # 沿轴切分的位置，第18列开始往后为y
                      axis=1)  #在这里就没有序号一说了
    print("Birch聚类算法程序运行时间：%.8s ",len(y1))

    y1.loc[y1['Y'] == 0] = 0  # 0代表70
    y1.loc[y1['Y'] == 1] = 1  # 2代表60

    x = x1.values[:, [3,15,8,2,17]].astype('float')  # 最完美的特征组合 3,15,8,14
    y = y1.values.astype('float')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,  # 所要划分的样本特征集
                                                                        y,  # 所要划分的样本结果
                                                                        random_state=2,  # 随机数种子确保产生的随机数组相同
                                                                        test_size=0.3)  # 测试样本占比
    # ********************Birch聚类*****************************
    # 创建Birch聚类
    ##########记录时间开始
    starttime = time.time()  # 开始计时

    birch= Birch(n_clusters=2, threshold = 0.05,branching_factor=500) #阈值很关键哦 0.05
    y_birch = birch.fit_predict(x)
    # print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(x, y_birch))
    # 用Calinski-Harabasz Index评估的聚类分数:
    # print(metrics.calinski_harabasz_score(x, y_birch))

    ##########记录时间结束
    endtime = time.time()
    dtime = endtime - starttime
    print("Birch聚类算法程序运行时间：%.8s s" % dtime)  # 显示到微秒

    # 想列出所有的判别和原来的y的信息的对比表
    r = np.concatenate((y_birch[:, np.newaxis], y), axis=1)  # 合并成一个
    k_dif = 0
    num = 0
    nums = []  # 记录分错的点index
    for i in r:
        num += 1
        if i[0] != i[1]:
            k_dif += 1
            nums.append(num)
    print("一共有", k_dif, "个点不同")

    # 画图一
    # cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['b', 'g', 'r'])

    figure(num)
    subplot(121)
    q = 0  # 记录选择的  特征向量
    p = 1
    fontsize_title = 40
    fontsize_axis = 40
    tick_size = 25
    legend_size = 25

    plt.scatter(x[y_birch == 0, q], x[y_birch == 0, p], s=100, c='red', label='Prediction of Bubble')
    plt.scatter(x[y_birch == 1, q], x[y_birch == 1, p], s=100, c='blue', label='Prediction of Particle')

    x_train_feature ='Feature 1', 'Feature 2','Max slope','I19_EPd3','I20_EPd4'
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Controids')
    plt.legend(loc='upper left',prop={'family': 'Times New Roman', 'size': legend_size})
    plt.tick_params(labelsize=tick_size)  # 坐标轴刻度数字大小
    plt.xlabel(x_train_feature[0],fontproperties = 'Times New Roman', fontsize=fontsize_axis)
    plt.ylabel(x_train_feature[1],fontproperties = 'Times New Roman', fontsize=fontsize_axis)
    plt.title('BIRCH used in data clustering',fontproperties = 'Times New Roman', fontsize=fontsize_title)

    for j in nums:
        plt.scatter(x[j - 1, q], x[j - 1, p], cmap=cm_dark, marker='x')

    subplot(122)
    plt.scatter(x[y.reshape(-1) == 0, q], x[y.reshape(-1) == 0, p], s=100, c='red', label='Bubble')
    plt.scatter(x[y.reshape(-1) == 1, q], x[y.reshape(-1) == 1, p], s=100, c='blue', label='Particle')
    plt.title('Real data classification',fontproperties = 'Times New Roman', fontsize=fontsize_title)
    plt.legend(loc='upper left',prop={'family': 'Times New Roman', 'size': legend_size})
    plt.tick_params(labelsize=tick_size)  # 坐标轴刻度数字大小

    plt.xlabel(x_train_feature[0], fontproperties = 'Times New Roman',fontsize=fontsize_axis)
    plt.ylabel(x_train_feature[1], fontproperties = 'Times New Roman',fontsize=fontsize_axis)

    #######################################################################
    # 画直方图  查看聚类分类后的结果
    x_real, y_real = np.split(file, (21,), axis=1)  # 读取为归一化的幅值数据
    fig, ax = plt.subplots()
    # the histogram of the data
    sqort_3 = []  # 原始真实混杂的颗粒分布
    for i in range(0, len(x1.values[:, q])):
        sqort_3.append(math.pow(x_real.values[i, q], 1 / 3) * 20.894 - 0.35)  # 找出对应关系

    x_birch0_sqort3 = []  # 记录k-means聚类为0的点的索引
    x_birch1_sqort3 = []
    birch_index_0 = np.where(y_birch == 0)[0]
    birch_index_1 = np.where(y_birch == 1)[0]

    for i in birch_index_0:
        x_birch0_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 - 0.35)

    for i in birch_index_1:
        x_birch1_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 - 0.35)

    subplot(131)
    plt.hist(x_birch0_sqort3, bins=25, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.title(u'x_birch0_data')

    subplot(132)
    plt.hist(x_birch1_sqort3, bins=25, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.title(u'x_birch1_data')

    subplot(133)
    plt.hist(sqort_3, bins=25, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.title(u'real_hunhe_data')

    # 绘制正确的分类数据分布图
    Real_index_0 = np.where(y1 == 0)[0]
    Real_index_1 = np.where(y1 == 1)[0]

    x_0_Real_sqort3 = []  # 记录k-means聚类为0的点的索引
    x_1_Real_sqort3 = []

    for i in Real_index_0:
        x_0_Real_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 +0.26)

    for i in Real_index_1:
        x_1_Real_sqort3.append(math.pow(x_real.values[i, 0], 1 / 3) * 20.894 +0.26)

    fig, ax = plt.subplots()
    subplot(121)
    plt.hist(x_0_Real_sqort3, bins=25, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.title(u'real_particle_data')

    subplot(122)
    plt.hist(x_1_Real_sqort3, bins=25, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.title(u'real_bubble_data')

    plt.show()

    return x_birch0_sqort3,x_birch1_sqort3

def birch_hist(file_birch,x,y):

    plt.hist(file_birch, bins=90, range=None, density=False,weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             # log=False, color="#FF6600", label="On-line ( Identified Bubble )", stacked=False)
             # log = False, color = "#FF6600", label = "On-line ( Particle + Removing Bubble )", stacked = False)
            log = False, color = "#FF6600", label = "On-line ( Particle + Bubble )", stacked = False)
             #log = False, color = "#FF6600",label = u"在线（颗粒 + 去除气泡）",   stacked = False)

    plt.bar(x, y, width=0.5, label="Off-line ( Particle )", color="#3366FF", alpha=0.85)

    font_size = 40  # 显示字体和坐标数字大小
    plt.legend(loc='upper right',prop={'family': 'Times New Roman', 'size': font_size})

    plt.xlabel("Diameter(μm)", fontproperties = 'Times New Roman', fontsize=font_size+5)
    plt.ylabel("Number", fontproperties = 'Times New Roman', fontsize=font_size+5)
    plt.xticks(fontsize=font_size-5)
    plt.yticks(fontsize=font_size-5)

    plt.show()
#******************************读取文件，划分数据集************************
## 读取原始-未归一化数据
if __name__ == "__main__":
    #读取数据
    data10 = pd.read_excel("D:/20thesis/826/data_813/Classification - Validation/hunhe_60-80_qipao/hunhe_813_200_60-80_qipao.xlsx", encoding='gdk') #80-qipao
    # 将数据 按照幅值分为两个部分 60 + 80
    x2,y2=np.split(data10,                                       #要切分的数组
                      (21,),                                       #沿轴切分的位置，第18列开始往后为y
                       axis=1)

    index_80 = x2.loc[x2['I1_mv']>35].index #存放索引
    index_60 = x2.loc[x2['I1_mv']<35].index #存放索引

    file1 = x2.iloc[index_60]  # 通过 iloc获得dataframe,再合并前面的特征 和后面的标签
    file2 = x2.iloc[index_80]

    Sd_file1=Standardized_data(file1).join(y2.iloc[index_60]) #标准化
    Sd_file2=Standardized_data(file2).join(y2.iloc[index_80]) #标准化

    # 聚类算法，并且绘制
    file1_birch0_D, file1_birch1_D = Clustering_deletion(Sd_file1,file1,1) #将归一化的数据，进行聚类分析，找到颗粒的索引，绘制
                                        #粒径分布图
    file2_birch0_D, file2_birch1_D = Clustering_deletion(Sd_file2,file2,2)

    #将分类结果进行组合
    file_birch0_D = file1_birch0_D + file2_birch0_D

    file_birch1_D = file1_birch1_D + file2_birch1_D

    file_birch2_D = file1_birch0_D + file2_birch1_D

    file_birch3_D = file1_birch1_D + file2_birch0_D

    file_birch_D = file1_birch0_D + file2_birch0_D + file1_birch1_D + file2_birch1_D

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

    #观察一下 在线数据 birch聚类的四个分部 和 贝克曼的对比结果。
    plt.figure(10)
    birch_hist(file_birch0_D,x3,y4)
    plt.figure(11)
    birch_hist(file_birch1_D,x3,y4)
    plt.figure(12)
    birch_hist(file_birch2_D,x3,y4)
    plt.figure(13)
    birch_hist(file_birch3_D,x3,y4)
    plt.figure(14)
    birch_hist(file_birch_D, x3, y4)#总体对比 原始在线 和 贝克曼 离线的

    plt.show()






















