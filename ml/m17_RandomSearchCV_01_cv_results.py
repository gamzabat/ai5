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
    {'C' : [1, 10, 100, 1000], 'kernel' : ['linear', 'sigmoid'], 'degree' : [3, 4, 5]},
    {'C' : [1, 10, 100], 'kernel' : ['rbf'], 'gamma' : [0.001, 0.0001]},
    {'C' : [1, 10, 100, 1000], 'kernel' : ['sigmoid'], 'gamma' : [0.01, 0.001, 0.0001], 'degree' : [3, 4]}
] # 54

# model
model = RandomizedSearchCV(SVC(), parameters, cv = kfold, verbose = 1, refit = True, n_jobs = -1)

start_time = time.time()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)

print('optimal parameter : ', model.best_estimator_) # optimal parame ter : SVC(C=1, kernel='linear')
print('optimal parameter : ', model.best_params_) # optimal parameter :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
print('best score : ', model.best_score_) # best score :  0.9833333333333334 - best score in train
print('model score : ', model.score(x_test, y_test)) # model score :  0.9333333333333333 - best score in test
print('accuracy score :', accuracy_score(y_test, y_predict)) # accuracy score : 0.9333333333333333
print('optimal tuning acc :', accuracy_score(y_test, y_pred_best)) # optimal tuning acc : 0.9333333333333333 <- use this
print('time taken :', time.time() - start_time) # time taken : 1.4998657703399658

print(pd.DataFrame(model.cv_results_).T)
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending = True))
print(pd.DataFrame(model.cv_results_).columns)

PATH = './_save/m15_GS_CV_01/'

pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending = True).to_csv(PATH + 'm15_RS_cv_results.csv')