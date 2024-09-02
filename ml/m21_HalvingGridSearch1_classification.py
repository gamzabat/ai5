import numpy as np
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')

# data
x, y = load_digits(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8,
    stratify = y
)

print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
print(x_test.shape, y_test.shape) # (360, 64) (360,)

n_splits = 5

kfold = StratifiedKFold(
    n_splits = n_splits,
    shuffle = True,
    random_state = 3333
)

parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1,  0.2, 0.3], 'max_depth' : [3, 4, 5, 6, 8]},
    # {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    # {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    # {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0]}
]

# model
model = HalvingGridSearchCV(
    xgb.XGBClassifier(
        tree_method = 'hist',
        device = 'cuda:0',
        n_estimators = 50,
    ),
    parameters,
    cv = kfold,
    verbose = 3,
    refit = True,
    random_state = 333,
    factor = 3.2,
    min_resources = 100,
    max_resources = 1437
)

start_time = time.time()

model.fit(x_train, y_train)

# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 159
# max_resources_: 1437
# aggressive_elimination: False
# factor: 3

# iter: 0
# n_candidates: 25
# n_resources: 159
# Fitting 5 folds for each of 25 candidates, totalling 125 fits

# iter: 1
# n_candidates: 9
# n_resources: 477
# Fitting 5 folds for each of 9 candidates, totalling 45 fits

# iter: 2
# n_candidates: 3
# n_resources: 1431
# Fitting 5 folds for each of 3 candidates, totalling 15 fits

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