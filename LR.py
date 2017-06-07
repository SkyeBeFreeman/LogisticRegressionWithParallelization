# coding: utf-8

from random import seed
from random import randrange
from math import sqrt
from scipy import sparse
import numpy as np
import datetime,time

# 数据列数 = 1 + 132

# 自定义异常
class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

train_x = []
train_y = []
# 加载训练集
print(time.strftime("%Y-%m-%d %H:%M:%S") + " 训练集读入中...", flush=True)
with open("train_data.txt", 'r') as file:
    cnt = 0
    for line in file:
        train_row_data = [0] * 132
        row_data = str(line).split()
        train_y.append(int(row_data[0]))
        for i in range(1,len(row_data)):
            key_value = row_data[i].split(":")
            key = int(key_value[0])
            value = float(key_value[1])
            if (key <= 131):
                train_row_data[key - 1] = value
        train_x.append(train_row_data)
        cnt += 1
        # if (cnt == 500):
        #     break
train_x = np.array(train_x)
train_y = np.array(train_y).reshape((-1,1))
print("train_x type: " + str(type(train_x)))
print("train_x shape: " + str(train_x.shape))
print("train_y type: " + str(type(train_y)))
print("train_y shape: " + str(train_y.shape))
print(time.strftime("%Y-%m-%d %H:%M:%S") + " 训练集读取完成", flush=True)

# 平均归一化训练集feature
x_min = train_x.min(0)
x_max = train_x.max(0)
x_diff = x_max - x_min
x = (x - x_min)/x_diff

# 一些参数的初始化
m = train_x.shape[0]
n = train_x.shape[1]
alpha = 0.1
threshold = 0.0000001
lmd = 0.1
step = 1
theta = np.random.random((n, 1))

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义矩阵点乘
def dotMultiply(X, Y):
    X_row = X.shape[0]
    X_col = X.shape[1]
    Y_row = Y.shape[0]
    Y_col = Y.shape[1]
    if X_col != Y_row:
        raise MyError("矩阵大小不适配")
    result = []
    for i in range(X_row):
        line = []
        for j in range(Y_col):
            ans = 0
            for a in range(X_col):
                ans += X[i][a] * Y[a][j]
            line.append(ans)
        result.append(line)
    result = np.array(result)
    return result.reshape((X_row, Y_col))

# 定义矩阵对应位置相乘
def myMultiply(X, Y):
    X_row = X.shape[0]
    X_col = X.shape[1]
    Y_row = Y.shape[0]
    Y_col = Y.shape[1]
    if X_col != Y_col or X_row != Y_row:
        raise MyError("矩阵大小不匹配")
    result = []
    for i in range(X_row):
        line = []
        for j in range(Y_col):
            line.append(X[i][j] * Y[i][j])
        result.append(line)
    result = np.array(result)
    return result.reshape((X_row, Y_col))

# 定义矩阵求和
def mySum(X):
    X_row = X.shape[0]
    ans = 0
    for i in range(X_row):
        ans += X[i][0]
    return ans

# 训练模型
cost = 0
change = 1
cnt = 0
while (change >= threshold):
    h = sigmoid(dotMultiply(train_x, theta))
    new_cost = mySum(())