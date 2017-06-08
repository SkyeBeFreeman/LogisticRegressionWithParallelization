# coding: utf-8

from random import seed, randrange, random
from math import log, exp, pow, fabs
from time import strftime

# 数据列数 = 1 + 132

# 自定义异常
class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

train_x = []
train_y = []
m = 0
n = 132
# 加载训练集
print(strftime("%Y-%m-%d %H:%M:%S") + " 训练集读入中...", flush=True)
with open("train_data.txt", 'r') as file:
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
        m += 1
        if (m == 100):
            break

print("train_x type: " + str(type(train_x)))
print("train_y type: " + str(type(train_y)))
print(strftime("%Y-%m-%d %H:%M:%S") + " 训练集读取完成", flush=True)

# 读取测试集
print(strftime("%Y-%m-%d %H:%M:%S") + " 测试集读入中...", flush=True)
test_m = 0
test_x = []
with open("test_data.txt", 'r') as file:
    test_id = set()
    for line in file:
        test_row_data = [0] * 132
        row_data = str(line).split()
        id = int(row_data[0])
        if not (id in test_id):
            test_id.add(id)
            for i in range(1,len(row_data)):
                key_value = row_data[i].split(":")
                key = int(key_value[0])
                value = float(key_value[1])
                test_row_data[key - 1] = value
            test_x.append(test_row_data)
            test_m += 1

print(test_m, flush=True)
print(strftime("%Y-%m-%d %H:%M:%S") + " 测试集读取完成", flush=True)

# 平均归一化训练集和测试集的feature
for i in range(n):
    x_min = train_x[0][i]
    x_max = train_x[0][i]
    for j in range(m):
        if (x_min > train_x[j][i]):
            x_min = train_x[j][i]
        if (x_max < train_x[j][i]):
            x_max = train_x[j][i]
    x_diff = (x_max - x_min) * 1.0
    if (x_diff == 0):
        x_diff = 1
    for j in range(m):
        train_x[j][i] = (train_x[j][i] - x_min) * 1.0 / x_diff
    for j in range(test_m):
        test_x[j][i] = (test_x[j][i] - x_min) * 1.0 / x_diff

# print(test_x[:20][:], flush=True)

# 一些参数的初始化
alpha = 0.1
threshold = 0.000001
lmd = 0
step = 1
# theta = [0.5] * n
theta = [random() for i in range(n)]

# sigmoid函数
def sigmoid(X, m):
    result = X
    for i in range(m):
        if X[i][0] > 100:
            result[i][0] = 1
        if X[i][0] < -100:
            result[i][0] = 0
        try:
            result[i][0] = 1 / (1 + exp(-X[i][0]))
        except Exception as e:
            print (e)
            print(X[i][0])
            print(test_x[i])
    return result

# 矩阵点乘
def dotMultiply(X, Y, m, n):
    X_row = m
    X_col = n
    Y_row = n
    Y_col = 1
    result = []
    for i in range(X_row):
        line = []
        for j in range(Y_col):
            ans = 0
            for a in range(X_col):
                ans += X[i][a] * Y[a][j]
            line.append(ans)
        result.append(line)
    return result

# theta转秩
def T(X):
    result = []
    for i in range(n):
        line = []
        line.append(X[i])
        result.append(line)
    return result

# 获取代价函数值
def getCost(m, n, lmd, h, theta):
    ans = 0
    for i in range(m):
        if train_y[i] == 1:
            ans += log(h[i][0]) * (-1.0) / m
        else:
            ans += log(1.0 - h[i][0]) * (-1.0) / m
    regularzilation = 0
    for i in range(n):
        regularzilation += pow(theta[i], 2)
    regularzilation *= lmd / (2 * m)
    return ans + regularzilation


# 训练模型
print(strftime("%Y-%m-%d %H:%M:%S") + " 开始训练", flush=True)
cost = 0
change = 1
cnt = 0
# while (change >= threshold):
for x in range(1000):
    h = sigmoid(dotMultiply(train_x, T(theta), m, n), m)
    # print(h)
    new_cost = getCost(m, n, lmd, h, theta)
    for i in range(n):
        gradient = 0
        for j in range(m):
            gradient += (h[j][0] - train_y[j]) * train_x[j][i]
        gradient -= lmd * theta[i]
        gradient *= ((- alpha) / m)
        theta[i] += gradient
    change = fabs(cost - new_cost)
    cost = new_cost
    cnt += 1
    if (cnt % step == 0):
        print('cost:', cost, flush=True)
print(strftime("%Y-%m-%d %H:%M:%S") + " 训练结束", flush=True)

# print(theta)
# mymax = 0
# for i in range(n):
#     if mymax < fabs(theta[i]):
#         mymax = fabs(theta[i])
# print(mymax)

print(strftime("%Y-%m-%d %H:%M:%S") + " 开始预测", flush=True)
h = sigmoid(dotMultiply(test_x, T(theta), test_m, n), test_m)
# print(h)
print(strftime("%Y-%m-%d %H:%M:%S") + " 预测结束", flush=True)

rounded = [round(x) for x in h]

with open('result.txt', 'w') as output:
    index = 0
    output.write('id,label\n')
    for i in rounded:
        Id, val = index, i
        output.write('{},{}\n'.format(Id, val))
        index += 1

print(strftime("%Y-%m-%d %H:%M:%S") + " 结束", flush=True)