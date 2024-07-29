import sklearn as sk

print(sk.__version__) # 0.24.2

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston

# data
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print("======================= MCP 출력 ==========================")
model = load_model('./_save/keras30_mcp/01-boston/k29_240726_173139_0894-23.4416.hdf5')

loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)