# coding: utf-8

from random import seed
from random import randrange
from math import sqrt
from scipy import sparse
import numpy as np
import datetime,time

# 数据列数 = 1 + 132

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

# 平均归一化
x_min = train_x.min(0)
x_max = train_x.max(0)
x_diff = x_max - x_min
x = (x - x_min)/x_diff

m = train_x.shape[0]
n = train_x.shape[1]
alpha = 0.1
threshold = 0.0000001
lmd = 0.1
step = 1
theta = np.random.random((n, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

