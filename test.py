import numpy as np

a = [[1, 5], [2, 5]]
print(a)
a = np.array(a)
print(a)
b = a.min(0)
print(b)
c = a.max(0)
print(c)
d = c - b
print(d)
d[d!=0] = 1
print(d)