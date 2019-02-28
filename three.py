# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:12:17 2019

@author: panpe
"""

from time import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def incomeclass(temp):
    bottom=[10,10,10,10,10,10]
    stageTop=[100,400,500,4000,15000]
#    income=Data[u'收入']
    incometemp=np.zeros((1, len(bottom)))[0]
    
    for i,j in enumerate(bottom):
        if i>=(len(stageTop)):
#            print('c',i)
            incometemp[i]= math.log(temp, j)#计算以c为底，b的对数:
            return incometemp
#            break
            pass
        if temp > stageTop[i]:
#            print('a',i)
            incometemp[i]= math.log(stageTop[i], j)#计算以c为底，b的对数:
            temp=temp-stageTop[i]
            continue

            pass
        else:
#            print('b',i)
            incometemp[i] = math.log(temp+1,j)
            return incometemp
#            break
            pass
        pass

    pass


if __name__ == "__main__":
    data=[]
    with open('crm_for_cluster_mini.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip().split()) # 把末尾的'\n'删掉
            pass
        pass
    a11=[i for i in data if len(i)==11]
    b12=[i for i in data if len(i)==12 ]
    c=[i for i in data if len(i)!=12 and len(i)!=11 ]
    name=[str(i) for i in range(12)]
    bb=pd.DataFrame(columns=name,data=b12)
    aa=pd.DataFrame(columns=a11[0],data=a11[1:])
    del bb['4']
    bb.columns=aa.columns
    data=pd.concat([aa,bb])
    name=data.columns.tolist()
    

    ###########################################################################
    # 对流量做统计展示
    #针对文字无法显示中文
    plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    for i in name[-5:]:
        # 读取的是字符，转换成float型
        data[i]=data[i].astype('float')
        # 大图小图
        sns.set(color_codes=True)
        fig=plt.figure()
        plt.grid(False)
        plt.annotate("", xytext=(200, 0.002) ,xy=(800, 0.004),\
                     arrowprops=dict(facecolor='r',shrink=0.001))
        sns.distplot(data[i])
        plt.xlabel(i,fontproperties="FangSong")
        ax2 = fig.add_axes([0.35, 0.35, 0.4, 0.4])  # inside axes 
        y=[i for i in data[i] if  i < 25000]
        ax2.hist(y, bins=30,density=True, facecolor='g')
        plt.xlabel("mini-plus",fontproperties="FangSong")
#        break
        pass
    
    datacode=[]
    for z in name[-5:]:
        IncomeOneHot=np.zeros((len(data), 6))
        
        for i,j in enumerate(data[z]):
            IncomeOneHot[i]=incomeclass(j)
    #        break
            pass
        tyname=[z+"_"+str(c) for c in range(6)]
        datacode.append(pd.DataFrame(IncomeOneHot,columns=tyname))
#        break
        pass
    datacode=pd.concat(datacode,axis=1)
    
    pca = PCA(n_components=0.99).fit(datacode)
    datacode_pca = pca.transform(datacode)
    ###########################################################################
    #聚类
    squer = []  
    nclusters = 10
    for i in range(nclusters):

        k_means = KMeans(init='k-means++', n_clusters=(i+1))
        k_means.fit(datacode_pca)
        squer.append(k_means.inertia_)
        
    #plot the data    
    plt.figure()
    plt.title('squer')
    plt.plot(squer)
    plt.show()
    
    k_means = KMeans(init='k-means++', n_clusters=(4))
    k_means.fit(datacode_pca)
    data["tag"]=k_means.predict(datacode_pca)
    
#    temp=[]
    mean=[]
    for i in range(4):
        at=data[data['tag']==i].iloc[:,-5:]
#        temp.append(data[data['tag']==i].iloc[:,-5:])
        mean.append(data[data['tag']==i].iloc[:,-5:-1].mean().mean())
        pass

    a=pd.Series(mean).rank()
    rankdict=dict(zip(a.index,a.values))
    
    # 分类tag越高，等级越高
    data.tag=data.tag.apply(lambda x:rankdict[x])

    data_transform = PCA(n_components=2).fit(datacode).transform(datacode)
    plt.figure()
    plt.scatter(data_transform[:, 0], data_transform[:, 1], c=data.tag)

    
    plt.title("original data")    














