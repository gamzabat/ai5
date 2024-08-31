import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# data
x, y = load_iris(return_X_y = True)

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

n_splits = 5

kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 333)

# model
model = SVC()

# train
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv = kfold
) # 기준 점수 확인

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)

print('cross_val_predict ACC :', acc)

# cross_val_predict ACC : 0.9333333333333333