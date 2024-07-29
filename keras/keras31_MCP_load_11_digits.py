import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 data
x, y = load_digits(return_X_y = True)

y = pd.get_dummies(y)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

model = load_model('./_save/keras30_mcp/11-digits/k30_240726_200749_0011-0.0683.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))