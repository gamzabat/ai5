import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score

import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)

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

# 최종점수 0.9666666666666667
# 정확도 0.9666666666666667
# [0.07570875 0.10196804 0.19451183 0.62781143]
# Trech = 0.076, n =   4, ACC: 96.67%
# Trech = 0.102, n =   3, ACC: 96.67%
# Trech = 0.195, n =   2, ACC: 96.67%
# Trech = 0.628, n =   1, ACC: 90.00%