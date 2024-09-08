import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score

import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2, 
    random_state=423, 
    stratify=y
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

# 최종점수 0.956140350877193
# 정확도 0.956140350877193
# [0.         0.         0.         0.00167528 0.00447645 0.00733686
#  0.00748172 0.00762305 0.00862877 0.01072983 0.01105577 0.01142481
#  0.01478048 0.01502065 0.01786744 0.01908214 0.01912484 0.01938069
#  0.02037933 0.02555194 0.02688929 0.02843894 0.03436768 0.05130163
#  0.05579945 0.05862024 0.07221626 0.09304451 0.10668615 0.25101578]
# Trech = 0.000, n =  30, ACC: 95.61%
# Trech = 0.000, n =  30, ACC: 95.61%
# Trech = 0.000, n =  30, ACC: 95.61%
# Trech = 0.002, n =  27, ACC: 95.61%
# Trech = 0.004, n =  26, ACC: 96.49%
# Trech = 0.007, n =  25, ACC: 95.61%
# Trech = 0.007, n =  24, ACC: 95.61%
# Trech = 0.008, n =  23, ACC: 95.61%
# Trech = 0.009, n =  22, ACC: 94.74%
# Trech = 0.011, n =  21, ACC: 97.37%
# Trech = 0.011, n =  20, ACC: 98.25%
# Trech = 0.011, n =  19, ACC: 96.49%
# Trech = 0.015, n =  18, ACC: 96.49%
# Trech = 0.015, n =  17, ACC: 97.37%
# Trech = 0.018, n =  16, ACC: 93.86%
# Trech = 0.019, n =  15, ACC: 94.74%
# Trech = 0.019, n =  14, ACC: 93.86%
# Trech = 0.019, n =  13, ACC: 93.86%
# Trech = 0.020, n =  12, ACC: 95.61%
# Trech = 0.026, n =  11, ACC: 93.86%
# Trech = 0.027, n =  10, ACC: 95.61%
# Trech = 0.028, n =   9, ACC: 92.98%
# Trech = 0.034, n =   8, ACC: 95.61%
# Trech = 0.051, n =   7, ACC: 93.86%
# Trech = 0.056, n =   6, ACC: 93.86%
# Trech = 0.059, n =   5, ACC: 95.61%
# Trech = 0.072, n =   4, ACC: 95.61%
# Trech = 0.093, n =   3, ACC: 94.74%
# Trech = 0.107, n =   2, ACC: 87.72%
# Trech = 0.251, n =   1, ACC: 87.72%