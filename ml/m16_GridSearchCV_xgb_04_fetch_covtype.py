import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# data
dataset = fetch_covtype()

x = pd.DataFrame(dataset.data).iloc[:, :13]

y = dataset.target

y = LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8,
    stratify = y
)

n_splits = 5

kfold = StratifiedKFold(
    n_splits = n_splits,
    shuffle = True,
    random_state = 3333
)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12], 'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10 ,12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]}
] # 54

# model
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv = kfold, verbose = True, refit = True, n_jobs = -1)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    verbose = False,
    eval_set = [(x_train, y_train), (x_test, y_test)]
)

y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)

print('optimal parameter : ', model.best_estimator_)
print('optimal parameter : ', model.best_params_)
print('best score : ', model.best_score_)
print('model score : ', model.score(x_test, y_test))
print('accuracy score :', accuracy_score(y_test, y_predict))
print('optimal tuning acc :', accuracy_score(y_test, y_pred_best))
print('time taken :', time.time() - start_time)