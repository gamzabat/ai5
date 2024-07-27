import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 data
dataset = fetch_covtype()

x = pd.DataFrame(dataset.data).iloc[:, :13]

y = pd.get_dummies(dataset.target)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

#2 model
print("======================= MCP 출력 ==========================")

model = load_model('./_save/keras30_mcp/10-fetch-covtype/k30_240726_200042_0056-0.3158.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))