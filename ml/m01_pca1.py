from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# data
datasets = load_iris()

x = datasets.data
y = datasets.target

print(x)
print(x.shape, y.shape)

# before scale after pca
scaler = StandardScaler()

x = scaler.fit_transform(x)

for item in range(len(x[0])):
    x_pca = x.copy()

    pca = PCA(n_components = (item + 1))

    x_pca = pca.fit_transform(x_pca)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca,
        y,
        train_size = 0.8,
        random_state = 4444,
        stratify = y
    )

    model = RandomForestClassifier(random_state = 7777)

    model.fit(x_train, y_train)

    # predict
    results = model.score(x_test, y_test)

    print(x.shape)
    print((item + 1), "인 경우 model.score :", results)

# (150, 4)
# model.score : 1.0

# (150, 3)
# model.score : 0.9666666666666667

# (150, 2)
# model.score : 0.9666666666666667

# (150, 1)
# model.score : 0.8333333333333334