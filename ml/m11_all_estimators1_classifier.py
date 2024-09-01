import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings

warnings.filterwarnings('ignore')

# data
x, y = load_iris(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8,
    stratify = y
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

all = all_estimators(
    type_filter = 'classifier',
#    type_filter = 'regressor',
)

print('all Algorithms :', all)
print('sk version :', sk.__version__)
print('all Algorithms :', len(all))

maxName = ''
maxAccuracy = 0

for name, model in all:
    try:
        # model
        model = model()

        # train
        model.fit(x_train, y_train)

        # evaluation
        acc = model.score(x_test, y_test)

        if acc > maxAccuracy:
            maxAccuracy = acc
            maxName = name

        print(name, "'s accuracy :", acc)
    except:
        print(name, ' Except')

print('max name :', maxName, 'max accuracy :', maxAccuracy)

# sk version : 1.1.3
# all Algorithms : 41
# AdaBoostClassifier 's accuracy : 0.9666666666666667
# BaggingClassifier 's accuracy : 0.9666666666666667
# BernoulliNB 's accuracy : 0.8666666666666667
# CalibratedClassifierCV 's accuracy : 0.9333333333333333
# CategoricalNB  Except
# ClassifierChain  Except
# ComplementNB  Except
# DecisionTreeClassifier 's accuracy : 0.9666666666666667
# DummyClassifier 's accuracy : 0.3333333333333333
# ExtraTreeClassifier 's accuracy : 0.9333333333333333
# ExtraTreesClassifier 's accuracy : 0.9666666666666667
# GaussianNB 's accuracy : 0.9666666666666667
# GaussianProcessClassifier 's accuracy : 0.9666666666666667
# GradientBoostingClassifier 's accuracy : 0.9666666666666667
# HistGradientBoostingClassifier 's accuracy : 0.9333333333333333
# KNeighborsClassifier 's accuracy : 0.9666666666666667
# LabelPropagation 's accuracy : 0.9666666666666667
# LabelSpreading 's accuracy : 0.9666666666666667
# LinearDiscriminantAnalysis 's accuracy : 1.0
# LinearSVC 's accuracy : 0.9666666666666667
# LogisticRegression 's accuracy : 1.0
# LogisticRegressionCV 's accuracy : 1.0
# MLPClassifier 's accuracy : 1.0
# MultiOutputClassifier  Except
# MultinomialNB  Except
# NearestCentroid 's accuracy : 0.9
# NuSVC 's accuracy : 0.9666666666666667
# OneVsOneClassifier  Except
# OneVsRestClassifier  Except
# OutputCodeClassifier  Except
# PassiveAggressiveClassifier 's accuracy : 0.9666666666666667
# Perceptron 's accuracy : 0.9333333333333333
# QuadraticDiscriminantAnalysis 's accuracy : 1.0
# RadiusNeighborsClassifier 's accuracy : 0.9666666666666667
# RandomForestClassifier 's accuracy : 0.9666666666666667
# RidgeClassifier 's accuracy : 0.8
# RidgeClassifierCV 's accuracy : 0.8
# SGDClassifier 's accuracy : 0.9666666666666667
# SVC 's accuracy : 0.9666666666666667
# StackingClassifier  Except
# VotingClassifier  Except

# max name : LinearDiscriminantAnalysis max accuracy : 1.0