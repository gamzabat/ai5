import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

x, y = load_digits(return_X_y = True)

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
    {'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12], 'min_samples_leaf' : [3, 10], 'learning_rate' : [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
    {'max_depth' : [6, 8, 10 ,12], 'min_samples_leaf' : [3, 5, 7, 10], 'learning_rate' : [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10], 'learning_rate' : [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
    {'min_samples_split' : [2, 3, 5, 10], 'learning_rate' : [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]}
] # 54

# model
model = RandomizedSearchCV(xgb.XGBClassifier(), parameters, cv = kfold, verbose = False, refit = True, n_jobs = -1)

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

# optimal parameter :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=12, max_leaves=None,
#               min_child_weight=None, min_samples_leaf=3, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=100,
#               n_jobs=None, num_parallel_tree=None, ...)
# optimal parameter :  {'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 12, 'learning_rate': 0.1}
# best score :  0.9505928184281842
# model score :  0.9777777777777777
# accuracy score : 0.9777777777777777
# optimal tuning acc : 0.9777777777777777
# time taken : 7.0681750774383545