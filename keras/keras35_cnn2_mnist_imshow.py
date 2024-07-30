import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist

(x_train, x_test), (y_train, y_test) = mnist.load_data()

print(x_train) # 가장자리만 표시되므로 요약정보에는 0만 나온다

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape) # (60000, 28, 28) (10000, 28, 28) -> 흑백인 경우 1인데 (60000, 28, 28, 1)이어야 하는데 생략되어있다 
print(x_test.shape, y_test.shape) # (60000,) (10000,)

print(np.unique(y_train, return_counts = True))

print(pd.value_counts(y_test))

import matplotlib.pyplot as plt

plt.imshow(x_train[7777], 'gray')
plt.show()