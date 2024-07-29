from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_diabetes

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
    random_state = 7777
)

print("======================= MCP 출력 ==========================")
model = load_model('./_save/keras30_mcp/03-diabetes/k30_240726_185812_0112-3709.1504.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)