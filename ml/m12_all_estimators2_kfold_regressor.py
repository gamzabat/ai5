import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings

warnings.filterwarnings('ignore')

# data
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 333)

all = all_estimators(
#    type_filter = 'classifier',
    type_filter = 'regressor',
)

print('all Algorithms :', all)
print('sk version :', sk.__version__)
print('all Algorithms :', len(all))

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

        print("===================", name, "===================")
        print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

        print('cross_val_predict ACC :', acc, '\n')
    except:
        print(name, ' Except\n')