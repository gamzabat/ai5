import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score

import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=423
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
model = XGBRegressor(
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

acc = r2_score(y_test, y_pre)

print("정확도", acc)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold = i, prefit = False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = XGBRegressor(
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

    score = r2_score(y_test, select_y_predict)

    print('Trech = %.3f, n = %3d, ACC: %.2f%%' %(i, select_x_train.shape[1], score * 100))

# 최종점수 0.3292057046570199
# 정확도 0.3292057046570199
# [0.04781655 0.06008002 0.08312377 0.08554597 0.08898104 0.10028013
#  0.10454215 0.10678589 0.13601585 0.18682864]
# Trech = 0.048, n =  10, ACC: 26.42%
# Trech = 0.060, n =   9, ACC: 29.77%
# Trech = 0.083, n =   8, ACC: 17.19%
# Trech = 0.086, n =   7, ACC: 31.06%
# Trech = 0.089, n =   6, ACC: 24.14%
# Trech = 0.100, n =   5, ACC: 26.14%
# Trech = 0.105, n =   4, ACC: 25.56%
# Trech = 0.107, n =   3, ACC: 10.67%
# Trech = 0.136, n =   2, ACC: 9.87%
# Trech = 0.187, n =   1, ACC: -13.41%