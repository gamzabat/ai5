학생csv = "jena_배누리.py"

path1 = "./_data/kaggle/jena/"
path2 = "./_save/keras55/"

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

datasets = pd.read_csv(path1 + "jena_cliamte_2009_2016.csv")

print(datasets)
print(datasets.shape)

y_정답 = datasets.iloc[-144:, 1]

print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col = 0)

print(학생꺼)

print(y_정답[:5])
print(학생꺼[:5])

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_정답, 학생꺼)

print("RMSE :", rmse)