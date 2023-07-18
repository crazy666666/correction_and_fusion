#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xlrd
from sklearn.utils import Bunch


def svr_model(X,y,kernel,Xtest,ytest,C = 6):

    ss_X = StandardScaler()             #标准化操作函数
    X_tra_train = ss_X.fit_transform(X)
    X_tra_test = ss_X.fit_transform(Xtest)

    ss_y = StandardScaler()
    y_tra_train = ss_y.fit_transform(y.reshape(-1,1))      #reshape(-1,1) 转换成一列, 此时shape为(m,1)
    y_tra_test = ss_y.fit_transform(ytest.reshape(-1,1))
    #y_tra_train,y_tra_test 的 shape (m,1)

    X_train = X_tra_train
    y_train = y_tra_train
    X_test = X_tra_test
    y_test = y_tra_test

    svr = SVR(C=C,kernel=kernel,gamma=0.475,tol=1e-3,epsilon=0.08)  #配置支持向量机

    #传入svr.fit()的参数需要 X_tra_train的shape为(m,n),y_tra_train的shape为(m,)
    svr.fit(X_train,y_train.ravel())   #fit一下，即相当于训练过程

    # svr_y_predict = svr.predict(X_test)  #预测X_tra_test，并得到
    svr_y_predict = svr.predict(X_test)  #预测X_tra_test，并得到

    predict = ss_y.inverse_transform(svr_y_predict.reshape(-1,1))  #进行反归一化操作，得到归一化之前的真实数据
    true = ss_y.inverse_transform(y_tra_test)
    # predict = svr_y_predict
    # print(predict.shape)
    # true = ytest
    # print(true.shape)
    # plt.plot(predict)
    # plt.plot(y)
    # plt.show()
    mse = mean_squared_error(true,predict)  #均方误差
    mae = mean_absolute_error(true,predict) #平均绝对误差
    R2 = r2_score(true,predict) #R2
    rmse = mse**0.5   #均方根误差
    mre = np.average(np.abs(true - predict)/true)/y_test.shape[0]  #平均相对误差
    mape = mean_absolute_percentage_error(true, predict)

    #1.输出结果
    # print("均方误差:",mse)
    print("平均绝对误差:",mae)
    print("均方根误差",rmse)
    # print("平均相对误差",mre)
    # print("R2",R2)
    # print("平均绝对百分比误差：",mape)
    # #2.画图
    # device_X = X_tra_test.shape[0]
    # X = np.linspace(0,1000,device_X)
    # plt.plot(X,true,'bo:')
    # plt.plot(X,predict,'r*-')
    # plt.title("rmse=%f mse=%f mae=%f mre=%f" % (rmse,mse,mae,mre))
    # plt.show()

hhh = xlrd.open_workbook(r"F:\datasets\pmdataset\yb1\littlepaper\data1.xls")
sh1 = hhh.sheet_by_index(3) # 合适照度
sh2 = hhh.sheet_by_index(4) # 低照度
sh3 = hhh.sheet_by_index(7) # 直方图均衡化
sh4 = hhh.sheet_by_index(6) # gamma
sh5 = hhh.sheet_by_index(8) # retinex
sh9 = hhh.sheet_by_index(5) # 本文算法8

X = np.zeros((1800,6))
for i in range(6):
    col_x = sh1.col_values(i,0,1800)
    X[:,i] = col_x
Y = np.zeros((1800,1))
col_y = sh1.col_values(6,0,1800)
Y[:,0] = col_y
Y = np.squeeze(Y)

X2 = np.zeros((1800,8))
for i in range(8):
    col_x2 = sh9.col_values(i,0,1800)
    X2[:,i] = col_x2
Y2 = np.zeros((1800,1))
col_y2 = sh9.col_values(8,0,1800)
Y2[:,0] = col_y2
Y2 = np.squeeze(Y2)

for i in range(10):
    # 合适照度训练，合适照度测试
    xtrain,_1,ytrain,_2 = train_test_split(X,Y,test_size=0.2,random_state=10)
    _3,xtest,_4,ytest = train_test_split(X,Y,test_size=0.2)

    # # 合适照度训练，低照度测试
    # xtrain,_1,ytrain,_2 = train_test_split(X,Y,test_size=0.2,random_state=10)
    # _3,xtest,_4,ytest = train_test_split(X2,Y2)

    # print('----------第{}次测试----------'.format(i+1))
    svr_model(xtrain,ytrain,kernel="rbf",Xtest=xtest,ytest=ytest)