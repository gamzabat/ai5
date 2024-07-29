import numpy as np
import pandas as pd

from sklearn.datasets import load_wine

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 data
dataset = load_wine()

x = dataset.data
y = pd.get_dummies(dataset.target)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 5555
)

print("======================= MCP 출력 ==========================")

model = load_model('./_save/keras30_mcp/09-wine/k30_240726_194637_0287-0.1073.hdf5')

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))