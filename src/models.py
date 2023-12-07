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

def fullPreProcessing(X_train, y_train):
    cleaned_data = X_train.drop(columns=['cursor_position','down_event', 'up_event','text_change']).groupby('id').sum() 
    mins_data = X_train.drop(columns=['cursor_position','down_event', 'up_event','text_change']).groupby("id").min()
    maxes_data = X_train.drop(columns=['cursor_position','down_event', 'up_event','text_change']).groupby("id").max()

    cleaned_data['min_down_time'] = mins_data['down_time']
    cleaned_data['max_up_time'] = maxes_data['up_time']
    cleaned_data['min_action_time'] = mins_data['action_time']
    cleaned_data['max_action_time'] = maxes_data['action_time']
    cleaned_data = cleaned_data.merge(X_train, on='id')

    drop_variables = ['id']
    passthrough_variables = ['score']
    scale_variables = ['event_id', 'down_time', 'up_time', 'action_time', 'word_count',
                    'min_down_time', 'max_up_time', 'min_action_time', 'max_action_time']
    preprocessor = make_column_transformer(
    ('drop', drop_variables),
    ("passthrough", passthrough_variables),
    (StandardScaler(), scale_variables)
                )
    transformed = preprocessor.fit_transform(cleaned_data)
    column_names = passthrough_variables + scale_variables
    X_trained_transformed = pd.DataFrame(transformed, columns=column_names)
    train_df, test_df = train_test_split(X_trained_transformed, test_size=0.2, random_state=42)
    y_train = train_df['score']
    X_train = train_df.drop(columns='score')
    y_test = test_df['score']
    X_test = test_df.drop(columns='score')

    return X_train, y_train, X_test, y_test

def makeBestRandomForest(X, y):
    best_forest = RandomForestRegressor(n_estimators=66, max_depth=5, random_state=42)
    best_forest.fit(X, y)
    return best_forest
