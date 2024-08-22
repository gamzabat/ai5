from tensorflow.keras.datasets import mnist

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis = 0)

x = x / 255.

print(np.min(x), np.max(x)) # 0.0 1.0

x = x.reshape(-1, x.shape[1] * x.shape[2])

pca = PCA(n_components = x.shape[1])

x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)

print(np.argmax(cumsum >= 1.0) + 1) # 713
print(np.argmax(cumsum >= 0.999) + 1) # 486
print(np.argmax(cumsum >= 0.99) + 1) # 331
print(np.argmax(cumsum >= 0.95) + 1) # 154

# 실습
# pca를 통해 0.95 이상인 n_components는 몇 개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일 때 몇 개