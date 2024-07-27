import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer

#1 data
datasets = load_breast_cancer()

print(datasets)

print(datasets.DESCR)
print(datasets.feature_names)

# y의 값이 0, 1이다

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 3333
)

# model
print("======================= MCP 출력 ==========================")
model = load_model('./_save/keras30_mcp/06-cancer/k30_240726_193325_0188-0.0884.hdf5')

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))