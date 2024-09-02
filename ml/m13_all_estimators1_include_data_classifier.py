import numpy as np
from sklearn.datasets import load_boston, load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings

import time

warnings.filterwarnings('ignore')

# data
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['iris', 'cancer', 'wine', 'digits']

all = all_estimators(
#    type_filter = 'classifier',
    type_filter = 'regressor',
)

print('all Algorithms :', all)
print('sk version :', sk.__version__)
print('all Algorithms :', len(all))

start_time = time.time()

for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle = True,
        random_state = 3333,
        train_size = 0.8,
        stratify = y
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    maxName = ''
    maxScoresMean = 0

    n_splits = 5

    kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 333)

    for name, model in all:
        try:
            # model
            model = model()

            # train
            model.fit(x_train, y_train)

            # evaluation
            acc = model.score(x_test, y_test)

            scores = cross_val_score(
                model,
                x,
                y,
                cv = kfold
            )

            scores_mean = np.round(np.mean(scores), 4)

            if scores_mean > maxScoresMean:
                maxScoresMean = scores_mean
                maxName = name
        except:
            pass

    print("===================", data_name[index], "===================")
    print("===================", maxName, "===================")
    print("===================", maxScoresMean, "===================\n")