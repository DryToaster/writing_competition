import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import xgboost as x
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim

def makeBestRandomForest(X, y):
    best_forest = RandomForestRegressor(n_estimators=66, max_depth=5, random_state=42)
    best_forest.fit(X, y)
    return best_forest

def modelSelectionProcess(X_train, y_train, X_test, y_test):
    models = {
    linear_model.LinearRegression(),
    linear_model.Ridge(),
    DecisionTreeRegressor(max_depth=15),
    RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42),
    svm.SVR(),
    KNeighborsRegressor(n_neighbors=13),
    AdaBoostRegressor(random_state=42),
    GradientBoostingRegressor(),
    x.XGBRegressor()
    }

    print("MSE:")
    for model in models:
        model.fit(X_train, y_train)
        print(type(model).__name__, 1 - model.score(X_test, y_test))


