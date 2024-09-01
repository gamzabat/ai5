import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings

warnings.filterwarnings('ignore')

# data
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

all = all_estimators(
#    type_filter = 'classifier',
    type_filter = 'regressor',
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
# all Algorithms : 55
# ARDRegression 's accuracy : 0.7383706445386875
# AdaBoostRegressor 's accuracy : 0.8306265360693887
# BaggingRegressor 's accuracy : 0.8630811232182196
# BayesianRidge 's accuracy : 0.7431365408199815
# CCA  Except
# DecisionTreeRegressor 's accuracy : 0.7180568832780219
# DummyRegressor 's accuracy : -0.006004869859544293
# ElasticNet 's accuracy : 0.7065018912404504
# ElasticNetCV 's accuracy : 0.7414827992265531
# ExtraTreeRegressor 's accuracy : 0.2363877657940746
# ExtraTreesRegressor 's accuracy : 0.8742224776221794
# GammaRegressor 's accuracy : 0.6677494988226644
# GaussianProcessRegressor 's accuracy : 0.12166155361275688
# GradientBoostingRegressor 's accuracy : 0.8870182736577423
# HistGradientBoostingRegressor 's accuracy : 0.8662732958956375
# HuberRegressor 's accuracy : 0.7669243680780524
# IsotonicRegression  Except
# KNeighborsRegressor 's accuracy : 0.7756366291304783
# KernelRidge 's accuracy : -5.495953823096072
# Lars 's accuracy : 0.7235498534957374
# LarsCV 's accuracy : 0.7362908289469872
# Lasso 's accuracy : 0.7097869425498822
# LassoCV 's accuracy : 0.7385437929766228
# LassoLars 's accuracy : -0.006004869859544293
# LassoLarsCV 's accuracy : 0.73904067822171
# LassoLarsIC 's accuracy : 0.7393027896405651
# LinearRegression 's accuracy : 0.7376162675112963
# LinearSVR 's accuracy : 0.7651647894010919
# MLPRegressor 's accuracy : 0.6972527787818484
# MultiOutputRegressor  Except
# MultiTaskElasticNet  Except
# MultiTaskElasticNetCV  Except
# MultiTaskLasso  Except
# MultiTaskLassoCV  Except
# NuSVR 's accuracy : 0.7042880162707572
# OrthogonalMatchingPursuit 's accuracy : 0.6026993461901169
# OrthogonalMatchingPursuitCV 's accuracy : 0.6967159748916218
# MultiOutputRegressor  Except
# MultiTaskElasticNet  Except
# MultiTaskElasticNetCV  Except
# MultiTaskLasso  Except
# MultiTaskLassoCV  Except
# NuSVR 's accuracy : 0.7042880162707572
# OrthogonalMatchingPursuit 's accuracy : 0.6026993461901169
# OrthogonalMatchingPursuitCV 's accuracy : 0.6967159748916218
# MultiTaskElasticNet  Except
# MultiTaskElasticNetCV  Except
# MultiTaskLasso  Except
# MultiTaskLassoCV  Except
# NuSVR 's accuracy : 0.7042880162707572
# OrthogonalMatchingPursuit 's accuracy : 0.6026993461901169
# OrthogonalMatchingPursuitCV 's accuracy : 0.6967159748916218
# MultiTaskLasso  Except
# MultiTaskLassoCV  Except
# NuSVR 's accuracy : 0.7042880162707572
# OrthogonalMatchingPursuit 's accuracy : 0.6026993461901169
# OrthogonalMatchingPursuitCV 's accuracy : 0.6967159748916218
# OrthogonalMatchingPursuit 's accuracy : 0.6026993461901169
# OrthogonalMatchingPursuitCV 's accuracy : 0.6967159748916218
# PLSCanonical  Except
# PLSRegression 's accuracy : 0.7629379698868057
# PassiveAggressiveRegressor 's accuracy : 0.6185969707227337
# PLSCanonical  Except
# PLSRegression 's accuracy : 0.7629379698868057
# PassiveAggressiveRegressor 's accuracy : 0.6185969707227337
# PoissonRegressor 's accuracy : 0.7950923769252675
# PoissonRegressor 's accuracy : 0.7950923769252675
# QuantileRegressor 's accuracy : -0.008760435670932543
# QuantileRegressor 's accuracy : -0.008760435670932543
# RANSACRegressor 's accuracy : 0.6520013848597382
# RadiusNeighborsRegressor  Except
# RandomForestRegressor 's accuracy : 0.8788395761780856
# RegressorChain  Except
# Ridge 's accuracy : 0.7387940933009151
# RidgeCV 's accuracy : 0.7387940933006649
# SGDRegressor 's accuracy : 0.7392083235182652
# SVR 's accuracy : 0.7172587790238967
# StackingRegressor  Except
# TheilSenRegressor 's accuracy : 0.5819709989494202
# TransformedTargetRegressor 's accuracy : 0.7376162675112963
# TweedieRegressor 's accuracy : 0.6986775336593775
# VotingRegressor  Except

# max name : GradientBoostingRegressor max accuracy : 0.8870182736577423