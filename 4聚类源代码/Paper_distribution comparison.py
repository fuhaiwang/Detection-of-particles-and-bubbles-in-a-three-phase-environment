# -*- coding:gb2312 -*- #��һ������ÿ��������f�ֲ���չʾ���߼��������ֲ� �� ���������߲�������ĶԱȡ�
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from pylab import *
import time

from pylab import *
import math
import xlrd

starttime = time.time()

def draw_keli(x1):
    # �����η�
    sqort_3 = []
    for i in range(0, len(x1.values[:, q])):
        # sqort_3.append(math.pow(x1.values[i, 0], 1 / 3) )
        sqort_3.append(math.pow(x1.values[i, q], 1/ 3 )* 20.894+0.26 )  # �ҳ���Ӧ��ϵ 110 90

    fig1, axs = plt.subplots(nrows=1, ncols=1)
    # n, bins, patches = axs.hist(sqort_3, bins=110, density=False, label="On-line ( Particle + Bubble )", facecolor='#FF6600',#110,
    #                             weights=[1. / len(sqort_3)] * len(sqort_3))  # �������
    n, bins, patches = axs.hist(sqort_3, bins=110, density=False, label='On-line ( Particle )',facecolor='#FF6600',  # 110,
                                weights=[1. / len(sqort_3)] * len(sqort_3))  # �������
    plt.legend(loc='best')
    fig1.tight_layout()
    return axs

if __name__ == "__main__":
    # plt.close('all')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # ******************************��ȡ�ļ����������ݼ�************************
    data = pd.read_excel("D:/20thesis/826/data_813/Classification - Validation/813_200_60-80.xlsx", encoding='gdk')
    # data = pd.read_excel("D:/20thesis/826/data_813/Classification - No Validation/813_200_60-80_qipao.xlsx", encoding='gdk') #���ݺ˿������߲���

    xuhao, x1, y1 = np.split(data,  # Ҫ�зֵ�����
                             (0, 20,),  # �����зֵ�λ�ã���18�п�ʼ����Ϊy
                             axis=1)
    print(x1.values[-1, 0])
    x = x1.values[:, [1, 1]].astype('float')  # 1.ֻѡ��2,8�������� 0 1 2 3 5 7 8 9  10 11
    ####################### ��ֱ��ͼ#################################################
    font_size = 40  # ��ʾ������������ִ�С
    q = 0
    fig, ax = plt.subplots()
    data1 = x1.values[:, q].astype('float')  # 0=70

    n, bins, patches = ax.hist([data1], bins=150, density=False, weights=[1. / len(data1)] * len(data1), label="Particle")  # �������
    fig.tight_layout()
    ###########  �� �����������#################################################
    def NumericOutlier(value):
        # value�ǵ�ά����ʷ����
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
    # �ڶ����� ������ �� �Ҳ�� �Ա�###################################
    axs=draw_keli(x1)

    ##  ��ͼ ��������ͼ ��һ��  #########################################
    xx3 = xlrd.open_workbook(r"D:\20thesis\826\data_813\beckman_200_60-80_2.xls")
    sheet1 = xx3.sheet_by_name("Sheet1")
    x3 = sheet1.col_values(0, 0, 86)
    y3 = sheet1.col_values(1, 0, 86)

    def Standardized_data(file):
        x = file
        Sd_file = np.array(x) / sum(x) *4/5  # ����һ�����������������㣬��Ҫ����һ�������ͱ�����ͳһ��
        return Sd_file

    y3 = Standardized_data(y3)

    axs.bar(x3, y3, width=0.5, label="Off-line ( Particle) ", color="#3366FF", alpha=0.85)

    # font_size = 22  # ��ʾ������������ִ�С
    plt.legend(prop={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel("Diameter(��m)", fontproperties = 'Times New Roman', fontsize=font_size)
    plt.ylabel("Number(%)", fontproperties = 'Times New Roman', fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()

    endtime = time.time()
    dtime = endtime - starttime
    print("��������ʱ�䣺%.8s s" % dtime)  #��ʾ��΢��
