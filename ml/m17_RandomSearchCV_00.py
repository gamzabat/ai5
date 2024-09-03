import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd

# data
x, y = load_iris(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8
)

n_splits = 5

kfold = StratifiedKFold(
    n_splits = n_splits,
    shuffle = True,
    random_state = 3333
)

parameters = [
    {'C' : [1, 10, 100, 1000], 'kernel' : ['linear', 'sigmoid'], 'degree' : [3, 4, 5]}, # 24
    {'C' : [1, 10, 100], 'kernel' : ['rbf'], 'gamma' : [0.001, 0.0001]}, # 6
    {'C' : [1, 10, 100, 1000], 'kernel' : ['sigmoid'], 'gamma' : [0.01, 0.001, 0.0001], 'degree' : [3, 4]} # 24
] # 54

# model
model = RandomizedSearchCV(
    SVC(),
    parameters,
    cv = kfold,
    verbose = 1,
    refit = True,
    n_iter = 9,
    random_state = 9999,
    n_jobs = -1)

start_time = time.time()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)

print('optimal parameter : ', model.best_estimator_)
print('optimal parameter : ', model.best_params_)
print('best score : ', model.best_score_)
print('model score : ', model.score(x_test, y_test))
print('accuracy score :', accuracy_score(y_test, y_predict))
print('optimal tuning acc :', accuracy_score(y_test, y_pred_best))
print('time taken :', time.time() - start_time)