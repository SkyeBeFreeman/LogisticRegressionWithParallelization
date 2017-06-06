# coding: utf-8

from random import seed
from random import randrange
from math import sqrt
from scipy import sparse
import numpy as np
import pandas as pd
import datetime,time

# 数据列数 = 1 + 133

train_x = []
train_y = []
# 加载训练集
print(time.strftime("%Y-%m-%d %H:%M:%S") + " 训练集读入中...", flush=True)
with open("train_data.txt", 'r') as file:
    cnt = 0
    for line in file:
        train_row_data = [0] * 133
        row_data = str(line).split()
        train_y.append(int(row_data[0]))
        for i in range(1,len(row_data)):
            key_value = row_data[i].split(":")
            key = int(key_value[0])
            value = float(key_value[1])
            if (key <= 132):
                train_row_data[key] = value
        train_x.append(train_row_data)
        cnt += 1
        # if (cnt == 500):
        #     break
train_x = np.array(train_x)
train_y = np.array(train_y)
print(time.strftime("%Y-%m-%d %H:%M:%S") + " 训练集读取完成", flush=True)
print(train_x)
print(train_y)
