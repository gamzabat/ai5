# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

PATH = "./_data/kaggle/playground-series-s4e1/" # 상대경로

le = LabelEncoder()

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

le.fit(train_csv['Geography'])

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])

le.fit(train_csv['Gender'])

train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [110023 rows x 1 columns]

train_csv.info() # CustomerId Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)

train_csv.info()

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis = 1)

test_csv.info()

x = train_csv.drop(['Exited'], axis = 1)

y = train_csv['Exited']

print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 7777
)

print("======================= MCP 출력 ==========================")

model = load_model('./_save/keras30_mcp/08-kaggle-bank/k30_240726_194218_0019-0.5084.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

submission_csv['Exited'] = np.round(y_submit, 1)

submission_csv.to_csv(PATH + "sample_submission_0726.csv")