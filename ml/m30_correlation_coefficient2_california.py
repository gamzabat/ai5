from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xgboost.plotting import plot_importance

import seaborn as sns

random_state = 7777

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns = datasets.feature_names)

df['Target'] = y

print(df)

print("correlation coefficient hitmap ------------------------------")
print(df.corr())

sns.heatmap(data = df.corr(),
            square = True,
            annot = True,
            cbar = True)

plt.show()