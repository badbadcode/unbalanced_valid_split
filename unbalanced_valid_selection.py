# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:09:07 2019
  假设我使用过采样解决不平衡问题，并从训练集切分出验证集进行调参，那么可选用的方案：

1.从原始训练集抽出验证集（标签比例1：1），对剩余的训练集数据进行过采样，再训练;
2.从原始训练集分层抽样，抽出验证集（标签比例1000：1），对剩余的训练集数据进行过采样，再训练;
3.对原始的训练集数据进行过采样，从过采样后的训练集抽出验证集（标签比例1：1），再训练;

按直觉来分析：
方案3，不行，这样验证集和训练集可能有重复；
方案2，由于验证集是非平衡的，调参又会让模型整个又倾向于非平衡时的学习能力，模型就又退步了；
如果上述判断没错的话，【方案1】应该是实际操作时的合适选择。

@author: Mingyue
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler



#数据集切分方式,保证dev数据量相同

#0:不进行过抽样,直接按原比例切分处理 50：0.006984215；100：0.01396843；200：0.02793686
def plan0(X,y):
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.006984215, random_state=0)#随机选择25%作为测试集，剩余作为训练集    
    return X_train,y_train,X_dev,y_dev

#1.从原始训练集抽出验证集（标签比例1：1），对剩余的训练集数据进行过采样，再训练;
def plan1(X,y):
    dev_num = 50 #每种标签50个样本 
    y = list(y.iloc[:,:].values)
    X = list(X.iloc[:,:].values)
    y_dev = []
    X_dev = []
    X_train = []
    y_train = []
    for i in range(len(y)):
        if len(y_dev)<=dev_num and y[i]==1:
            y_dev.append(y[i])
            X_dev.append(X[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i]) 
    
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)    
    return X_resampled, y_resampled,X_dev,y_dev

#2.从原始训练集分层抽样，抽出验证集，对剩余的训练集数据进行过采样，再训练;50：0.006984215；100：0.01396843；200：0.02793686
def plan2(X,y):
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.006984215, random_state=0)#随机选择作为测试集，剩余作为训练集    
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    return X_resampled,y_resampled,X_dev,y_dev

#3.对原始的训练集数据进行过采样，从过采样后的训练集抽出验证集（标签比例1：1），再训练;50：0.00386011；100：0.00772022；200：0.01544044
def plan3(X,y):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    X_train,X_dev,y_train,y_dev = train_test_split(X_resampled, y_resampled, test_size=0.00386011, random_state=0)
    return X_train,y_train,X_dev,y_dev



#调参
def best_parameter(X_train,y_train,X_dev,y_dev):
    Cs=[0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,
         0.05,0.06,0.07,0.08,0.085,0.09,0.1]
    acc = 0 
    
    for c in Cs:        
        lr_penalty=LogisticRegression(C = c)
        lr_penalty.fit(X_train,y_train)
        predicted = lr_penalty.predict(X_dev)  
        acc_new = accuracy_score(y_dev,predicted)
        if acc < acc_new:
            acc = acc_new
            model = lr_penalty
            best_c = c
    return model,best_c

#训练数据，3个分类模型：逻辑回归-L2正则；神经网络-L2正则；SVM-L2正则
def Compare(X_train,y_train,X_test,y_test,plan):
    funcdic = {0:plan0,1:plan1,2:plan2,3:plan3}
    #plan对应的切分方案
    X_train,y_train,X_dev,y_dev = funcdic[plan](X_train,y_train)
    model,best_c = best_parameter(X_train,y_train,X_dev,y_dev)
    predicted = model.predict(X_test)  
    acc = accuracy_score(y_test,predicted)
    f1 = f1_score(y_test,predicted)
    return acc,f1,best_c


    
#1:数据处理： 切分train和test（8：2）
os.chdir(r'C:\Users\Mingyue\Desktop\Codes\unbalance')
data = pd.read_csv('HTRU_2.csv',header = None)

#给数据命【列名】
data.columns = ['X1','X2','X3','X4','X5','X6','X7','X8','y']
feature_names = ['X1','X2','X3','X4','X5','X6','X7','X8']
target_names = ['y']

X = data[feature_names]
y = data[target_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#随机选择25%作为测试集，剩余作为训练集


#2: 选择一种切分train-train,train-dev的方式（7：3），分别进行调参,得到模型的参数结果 （9个结果）
for i in range(4):
    acc,f1,best_c = Compare(X_train,y_train,X_test,y_test,i)
    print(i,'|准确率：',acc,'f1:',f1,'|best c:',best_c)


