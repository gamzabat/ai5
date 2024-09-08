import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score

import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 423,
    stratify = y
)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

es = xgb.callback.EarlyStopping(
    rounds = 50,
#    metric_name = 'logloss',
    data_name = 'validation_0',
    save_best = True
)

#2. 모델
model = XGBClassifier(
    n_estimators = 100,
    max_depth= 6, 
    gamma = 0,
    min_child_weigt = 0,
    subsample = 0.4,
#    eval_metric = 'logloss',
    reg_alpha = 0, #L1규제, 절대값
    reg_lamda = 1, #L2규제, 제곱합
    random_state = 3377,
    callbacks = [es]
)

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)], 
          verbose = 1)

#4. 평가 및 예측
result = model.score(x_test, y_test)

print("최종점수", result)

y_pre = model.predict(x_test)

acc = accuracy_score(y_test, y_pre)

print("정확도", acc)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold = i, prefit = False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = XGBClassifier(
        n_estimators = 100,
        max_depth= 6, 
        gamma = 0,
        min_child_weigt = 0,
        subsample = 0.4,
        # eval_metric = 'logloss',
        reg_alpha = 0, #L1규제, 절대값
        # reg_lamda = 1, #L2규제, 제곱합
        random_state = 3377,
#        callbacks = [es]
    )

    select_model.fit(
        select_x_train,
        y_train,
        eval_set = [(select_x_test, y_test)],
        verbose = 0
    )

    select_y_predict = select_model.predict(select_x_test)

    score = accuracy_score(y_test, select_y_predict)

    print('Trech = %.3f, n = %3d, ACC: %.2f%%' %(i, select_x_train.shape[1], score * 100))

# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.00318422 0.00363904 0.00407636 0.0047078
#  0.00476698 0.0051679  0.00569017 0.0058441  0.00654052 0.00667399
#  0.00687134 0.00692358 0.0075701  0.00788469 0.00823668 0.00856864
#  0.009089   0.00988046 0.00997189 0.01005749 0.01045881 0.01053532
#  0.01105781 0.01224048 0.01316231 0.01321586 0.01350575 0.01485419
#  0.01665243 0.01678191 0.01693977 0.01694941 0.01736692 0.01950199
#  0.0214899  0.02365862 0.02582922 0.03033383 0.03179514 0.03221823
#  0.03260894 0.03435527 0.03512405 0.03521014 0.04648387 0.04717391
#  0.05340942 0.05893938 0.06051221 0.09228989]
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.000, n =  64, ACC: 95.56%
# Trech = 0.003, n =  50, ACC: 95.56%
# Trech = 0.004, n =  49, ACC: 95.56%
# Trech = 0.004, n =  48, ACC: 95.56%
# Trech = 0.005, n =  47, ACC: 95.83%
# Trech = 0.005, n =  46, ACC: 95.00%
# Trech = 0.005, n =  45, ACC: 94.72%
# Trech = 0.006, n =  44, ACC: 95.56%
# Trech = 0.006, n =  43, ACC: 95.56%
# Trech = 0.007, n =  42, ACC: 95.83%
# Trech = 0.007, n =  41, ACC: 96.39%
# Trech = 0.007, n =  40, ACC: 94.72%
# Trech = 0.007, n =  39, ACC: 95.00%
# Trech = 0.008, n =  38, ACC: 95.28%
# Trech = 0.008, n =  37, ACC: 95.28%
# Trech = 0.008, n =  36, ACC: 95.00%
# Trech = 0.009, n =  35, ACC: 95.00%
# Trech = 0.009, n =  34, ACC: 94.44%
# Trech = 0.010, n =  33, ACC: 93.61%
# Trech = 0.010, n =  32, ACC: 94.17%
# Trech = 0.010, n =  31, ACC: 94.44%
# Trech = 0.010, n =  30, ACC: 94.17%
# Trech = 0.011, n =  29, ACC: 94.17%
# Trech = 0.011, n =  28, ACC: 93.89%
# Trech = 0.012, n =  27, ACC: 93.06%
# Trech = 0.013, n =  26, ACC: 93.33%
# Trech = 0.013, n =  25, ACC: 94.17%
# Trech = 0.014, n =  24, ACC: 93.33%
# Trech = 0.015, n =  23, ACC: 93.33%
# Trech = 0.017, n =  22, ACC: 93.06%
# Trech = 0.017, n =  21, ACC: 92.50%
# Trech = 0.017, n =  20, ACC: 92.50%
# Trech = 0.017, n =  19, ACC: 92.22%
# Trech = 0.017, n =  18, ACC: 90.28%
# Trech = 0.020, n =  17, ACC: 90.83%
# Trech = 0.021, n =  16, ACC: 90.00%
# Trech = 0.024, n =  15, ACC: 89.72%
# Trech = 0.026, n =  14, ACC: 88.89%
# Trech = 0.030, n =  13, ACC: 88.89%
# Trech = 0.032, n =  12, ACC: 90.56%
# Trech = 0.032, n =  11, ACC: 86.94%
# Trech = 0.033, n =  10, ACC: 83.61%
# Trech = 0.034, n =   9, ACC: 84.72%
# Trech = 0.035, n =   8, ACC: 80.56%
# Trech = 0.035, n =   7, ACC: 73.06%
# Trech = 0.046, n =   6, ACC: 66.11%
# Trech = 0.047, n =   5, ACC: 55.28%
# Trech = 0.053, n =   4, ACC: 52.22%
# Trech = 0.059, n =   3, ACC: 36.67%
# Trech = 0.061, n =   2, ACC: 30.00%
# Trech = 0.092, n =   1, ACC: 14.72%