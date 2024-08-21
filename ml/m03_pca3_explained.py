from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import numpy as np

# data
datasets = load_iris()

x = datasets.data
y = datasets.target

print(x)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 4444,
    stratify = y
)

# before scale after pca
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for item in reversed(range(len(x_train[0] - 1))):
    pca = PCA(n_components = (item + 1))

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    print(x_train.shape, x_test.shape)

    model = RandomForestClassifier(random_state = 7777)

    model.fit(x_train, y_train)

    # predict
    results = model.score(x_test, y_test)

    print(x_train.shape)
    print(x_test.shape)
    print((item + 1), "인 경우 model.score :", results)

# (120, 4)
# (30, 4)
# model.score : 0.9666666666666667

# (120, 3)
# (30, 3)
# model.score : 0.9666666666666667

# (120, 2)
# (30, 2)
# model.score : 0.9333333333333333

# (120, 1)
# (30, 1)
# model.score : 0.9333333333333333

# evr = pca.explained_variance_ratio_ # 설명가능한 변화율

# print(evr)
# print(np.sum(evr))

# evr_cumsum = np.cumsum(evr)

# print(evr_cumsum)

# import matplotlib.pyplot as plt

# plt.plot(evr_cumsum)
# plt.grid()
# plt.show()