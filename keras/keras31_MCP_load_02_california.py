from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing

import time

#1 data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7755
)

print("======================= MCP 출력 ==========================")
model = load_model('./_save/keras30_mcp/02-california/k30_240726_174654_0038-0.6518.hdf5')

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)