# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 data
PATH = "./_data/kaggle/santander/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

#y = pd.get_dummies(train_csv['target'])
y = train_csv['target']

print(y)

# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.95,
    random_state = 7777,
    stratify = y
)

model = load_model('./_save/keras30_mcp/12-kaggle-santander/k30_240726_201154_0007-0.2479.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

print(y_submit[:10])

# print(np.round(y_submit[:, 1]))

sample_submission_csv['target'] = np.round(y_submit).astype("int")

sample_submission_csv.to_csv(PATH + "sample_submission_0725.csv")