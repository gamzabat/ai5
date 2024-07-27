import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

data = np.array([[1, 2, 3, 1],
                 [4, 5, 6, 2],
                 [7, 8, 9, 3],
                 [10, 11, 12, 114],
                 [13, 14, 15, 115]])

#1 평균
means = np.mean(data, axis = 0)

print("평균 :", means) # 평균 : [ 7.  8.  9. 47.]

#2 모집단 분산 (나누기 n), ddof = 0 (default)
population_variances = np.var(data, axis = 0, )

print("모집단 분산 :", population_variances) # 분산  [  18.   18.   18. 3038.]

#3 표본분산 (나누기 n - 1), ddof = 1
variances = np.var(data, axis = 0, ddof = 1)

print("표본 분산 :", variances) # 표본 분산  [  22.5   22.5   22.5 3797.5]

#4 표본 표준편차
std = np.std(data, axis = 0, ddof = 1)

print("표본 표준편차 :", std) # 표본 표준편차 : [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5 StandardScaler
scaler = StandardScaler(with_std = False)

scaled_data = scaler.fit_transform(data)

print("StandardScaler : \n", scaled_data)