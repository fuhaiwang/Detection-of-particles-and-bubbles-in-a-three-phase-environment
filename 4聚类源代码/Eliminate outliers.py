#数据清洗    主要读取数据，   通过调节42行，q列标，进行3sigma的判别。筛选数据样本，数据清洗。
import numpy as np   #第一步分析每个特征的f分布
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection

from pylab import *
import math
import xlrd

def draw_keli(x1):
    # 开三次方
    sqort_3 = []
    for i in range(0, len(x1.values[:, q])):
        # sqort_3.append(math.pow(x1.values[i, 0], 1 / 3) )
        sqort_3.append(math.pow(x1.values[i, q], 1/ 3 )* 20.894+0.26 )  # 找出对应关系 110 90
    #适用于 仅含有颗粒的
    # fig1, axs = plt.subplots(nrows=1, ncols=1)
    # n, bins, patches = axs.hist(sqort_3, bins=90, density=False, label="On-line ( Particle )", facecolor='#3366FF',#110,
    #                              weights=[1. / len(sqort_3)] * len(sqort_3))  # 存放数据
    # 适用于带有气泡的
    fig1, axs = plt.subplots(nrows=1, ncols=1)
    n, bins, patches = axs.hist(sqort_3, bins=110, density=False, label="On-line ( Particle + Bubblet )", facecolor='#3366FF',#110,
                                 weights=[1. / len(sqort_3)] * len(sqort_3))  # 存放数据
    # fig1.tight_layout()
    return axs

# def draw_Bubblet(x1):
#     sqort_3 = []
#     for i in range(0, len(x1.values[:, q])):
#         # sqort_3.append(math.pow(x1.values[i, 0], 1 / 3) )
#         sqort_3.append(math.pow(x1.values[i, q], 1 ))  # 找出对应关系 * 20.894+0.26
#
#     fig1, axs = plt.subplots(nrows=1, ncols=1)
#     n, bins, patches = axs.hist(sqort_3, bins=150, density=False, label="On-line + Bubblet", facecolor='#FF8C00')
#                                 # weights=[1. / len(sqort_3)] * len(sqort_3))  # 存放数据
#     fig1.tight_layout()
#     return axs


if __name__ == "__main__":
    # 对每一列的特征进行判别。  筛除掉不符合的特征数据样本。
    q = 1  # 0-1  修改q，进行3sigma的判别
    # ******************************读取文件，划分数据集************************
    #读取连续无气泡 颗粒的数据分布
    # data = pd.read_excel("I:/anaconda/data_813/Classification - Validation/813_200_60-80.xlsx", encoding='gdk')
    # 读取连续气泡干扰下的 颗粒的数据分布
    # data = pd.read_excel("D:/20thesis/826/data_813/Classification - No Validation/813_200_80_qipao.xlsx",encoding='gdk')
    data = pd.read_excel("D:/20thesis/826/data_126/Classification - Validation/hunhe_60-80_qipao/hunhe_813_200_60-80_qipao4.xlsx",encoding='gdk')
    # data = pd.read_excel("D:/20thesis/826/data_126/Classification - No Validation/813_200_60-80_qipao.xlsx", encoding='gdk')
    # 读取人造 无气泡 颗粒的数据分布
    # data = pd.read_excel("I:/anaconda/data_813/Classification - Validation/hunhe_60-80_qipao/hunhe_813_200_60-80_qipao.xlsx", encoding='gdk')

    xuhao, x1, y1 = np.split(data,  # 要切分的数组
                             (0, 20,),  # 沿轴切分的位置，第18列开始往后为y
                             axis=1)
    print(x1.values[-1, 0])

    x = x1.values[:, [1, 1]].astype('float')  # 1.只选了2,8两个特征 0 1 2 3 5 7 8 9  10 11
    # y = y1.values.astype('float')

    #######################################################################
    # 画直方图
    font_size = 22  # 显示字体和坐标数字大小


    fig, ax = plt.subplots()
    data1 = x1.values[:, q].astype('float')  # 0=70

    n, bins, patches = ax.hist([data1], bins=150, density=False, weights=[1. / len(data1)] * len(data1), label="Particle")  # 存放数据
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    plt.show()

    #### 二 ####################################################
    data4 = pd.read_excel("D:/20thesis/826/data_126/Classification - Validation/813_200_60-80.xlsx", encoding='gdk')

    xuhao4, x4, y4 = np.split(data4,  # 要切分的数组
                              (0, 20,),  # 沿轴切分的位置，第18列开始往后为y
                              axis=1)
    print(x4.values[-1, 0])

    x = x4.values[:, [1, 1]].astype('float')  # 1.只选了2,8两个特征 0 1 2 3 5 7 8 9  10 11
    # y = y4.values.astype('float')

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x,  # 所要划分的样本特征集
                                                                        # y,  # 所要划分的样本结果
                                                                        # random_state=2,  # 随机数种子确保产生的随机数组相同
                                                                        # test_size=0.3)  # 测试样本占比


    #######################################################################
    # 画直方图
    # fig, ax = plt.subplots()

    data5 = x4.values[:,q].astype('float')  #0=70

    n, bins, patches = ax.hist([data5], bins=150,density =False,weights= [1./ len(data5)] * len(data5), label="Bubblet")  #存放数据

    # Tweak spacing to prevent clipping of ylabel
    # fig4.tight_layout()

    #  三 输出正常区间###########################################################

    def NumericOutlier(value):
        # value是单维的历史数据
        iqr = np.quantile(value, 0.85) - np.quantile(value, 0.25)
        quan_down = np.quantile(value, 0.25) - 1.5 * iqr
        quan_up = np.quantile(value, 0.85) + 1.5 * iqr
        return [float(quan_down), float(quan_up)]


    x2 = x1.values[:, q]
    [quan_down, quan_up] = NumericOutlier(x2)

    print([quan_down, quan_up])
    print(np.argwhere(x2 > quan_up), np.argwhere(x2 < quan_down))
    plt.legend(fontsize=font_size)
    plt.show()
    # 第二部分 贝克曼 和 我测得 对比###################################
    # 绘制我的图。归一化

    # # 开三次方
    # sqort_3 = []
    # for i in range(0, len(x1.values[:, q])):
    #     # sqort_3.append(math.pow(x1.values[i, 0], 1 / 3) )
    #     sqort_3.append(math.pow(x1.values[i, q], 1 / 3) * 20.894 - 0.35)  # 找出对应关系
    #
    # fig1, axs = plt.subplots(nrows=1, ncols=1)
    # n, bins, patches = axs.hist(sqort_3, bins=60, density=False, label="On-line + Bubblet", facecolor='#FF8C00',
    #                             weights=[1. / len(sqort_3)] * len(sqort_3))  # 存放数据
    # # Tweak spacing to prevent clipping of ylabel
    # fig1.tight_layout()
    axs=draw_keli(x1)


    ##  绘图 贝克曼的图 归一化  #########################################
    xx3 = xlrd.open_workbook(r"D:\20thesis\826\data_813\beckman_200_60-80_2.xls")
    sheet1 = xx3.sheet_by_name("Sheet1")
    x3 = sheet1.col_values(0, 0, 86)
    y3 = sheet1.col_values(1, 0, 86)


    def Standardized_data(file):
        x = file
        Sd_file = np.array(x) / sum(x)  # 方法一
        return Sd_file


    y3 = Standardized_data(y3)

    axs.bar(x3, y3, width=0.5, label="Off-line ( Particle ) ", color="#87CEFA", alpha=0.85)

    font_size = 22  # 显示字体和坐标数字大小
    plt.legend(fontsize=font_size)
    plt.xlabel("Diameter(μm)", fontsize=font_size)
    plt.ylabel("Number(%)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)


    # draw_Bubblet():

    plt.show()
