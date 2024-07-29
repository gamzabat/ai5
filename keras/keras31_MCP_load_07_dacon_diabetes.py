# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PATH = "./_data/dacon/diabetes/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

print(test_csv.isna().sum())

train_csv.info() # Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome

test_csv.info()

print(train_csv.shape)

train_csv = train_csv[train_csv['BloodPressure'] != 0]
train_csv = train_csv[train_csv['BMI'] > 0.0]
train_csv = train_csv[train_csv['Glucose'] > 0]
 
print(train_csv['Glucose'].value_counts())

print(train_csv.shape)

train_csv.info()

x = train_csv.drop(['Outcome'], axis = 1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 3333
)

print("======================= MCP 출력 ==========================")

model = load_model('./_save/keras30_mcp/07-dacon-diabetes/k30_240726_193802_0032-0.5639.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

submission_csv['Outcome'] = np.round(y_submit)

submission_csv.to_csv(PATH + "sample_submission_0725.csv")