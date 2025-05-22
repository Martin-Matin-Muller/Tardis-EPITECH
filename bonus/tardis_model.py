#!/usr/bin/env python
# coding: utf-8

# Building a prediction model is the third step of the Tardis Project. The goal is to use the values in our cleaned dataset to predict future delays.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.dummy import DummyRegressor
from datetime import datetime


categorical_features = ["Month", "Departure station", "Arrival station"]



def build_pipeline(categorical_features):
    '''numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])'''

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        #('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DummyRegressor())
    ])

    return pipe


def parse_params(param_str):
    if not param_str or param_str == "{}":
        return {}
    param_str = param_str.strip("{}")
    items = param_str.split(", ")
    param_dict = {}
    for item in items:
        key, value = item.split(": ", 1)
        key = key.strip("'")

        if key.startswith("regressor__"):
            key = key.replace("regressor__", "")

        if value == 'None':
            value = None
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                value = value.strip("'")
        param_dict[key] = value
    return param_dict


def predict_delay(date_str, departure, destination, target, df, results_df):
    df = df.copy()
    if df['Month'].dtype == object:
        month = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B")
    else:
        month = datetime.strptime(date_str, "%Y-%m-%d").month

    X = df[['Month', 'Departure station', 'Arrival station']]
    y = df[target]

    row = results_df[results_df['target'] == target].sort_values('rmse').iloc[0]
    model_name = row['model']
    best_params = parse_params(row['best_params'])

    model_map = {
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
        'LinearRegression': LinearRegression,
    }
    model_cls = model_map[model_name]

    pipe = build_pipeline(categorical_features)
    pipe.set_params(regressor=model_cls(**best_params))
    pipe.fit(X, y)

    input_data = pd.DataFrame([{
        'Month': month,
        'Departure station': departure,
        'Arrival station': destination
    }])

    # Predict
    prediction = pipe.predict(input_data)[0]
    return prediction
