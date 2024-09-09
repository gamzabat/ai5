from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xgboost.plotting import plot_importance

import seaborn as sns
import matplotlib

random_state = 7777

datasets = load_iris()

x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns = datasets.feature_names)

df['Target'] = y

print(df)

print("correlation coefficient hitmap ------------------------------")
print(df.corr())

sns.set(font_scale = 1.2)

sns.clustermap(data = df.corr(),
            square = True,
            annot = True,
            cbar = True)

plt.show()