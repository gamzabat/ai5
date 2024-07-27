import sklearn as sk

print(sk.__version__) # 0.24.2

from keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston

# data
dataset = load_boston()

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape) # (506, 13)

print(y)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

# model

print("======================= 1. save.model 출력 ==========================")
model = load_model('./_save/keras29_mcp/keras29_3_save_model.h5')

loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)
print("======================= 2. MCP 출력 ==========================")
model2 = load_model('./_save/keras29_mcp/keras29_mcp3.hdf5')

loss2 = model2.evaluate(x_test, y_test, verbose = 0)

y_predict = model2.predict(x_test)

r22 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r22)

# loss : 41.99968338012695
# r2 : 0.5187676571246922

# loss : 41.99968338012695
# r2 : 0.5187676571246922