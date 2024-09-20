import sys
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import ast
import re
from sklearn.utils import resample
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import date

def predict_with_interval(model, X_new, percentile=95):
    point_prediction = model.predict(X_new)
    X_new_processed = model[:-1].transform(X_new)
    tree_predictions = np.array([tree.predict(X_new_processed) for tree in model['regressor'].estimators_])

    lower_bound = np.percentile(tree_predictions, (100 - percentile) / 2., axis=0)
    upper_bound = np.percentile(tree_predictions, 100 - (100 - percentile) / 2., axis=0)

    # return point_prediction, point_prediction, point_prediction
    return point_prediction, lower_bound, upper_bound

def predict(model, title):
    current_date_formatted = datetime.now().strftime("%b %d, %Y")
    data = pd.DataFrame({
        'title': [title],
        'date': [current_date_formatted]  # Wrap in list to match dimensions
    })
    X = data[['title', 'date']]
    loaded_objects = joblib.load(model)
    model = loaded_objects['model']
    scaler = loaded_objects['scaler']
    point_prediction, lower_bound, upper_bound = predict_with_interval(model, X)
    high = scaler.inverse_transform(upper_bound.reshape(1, -1))[0][0]
    avg = scaler.inverse_transform(point_prediction.reshape(1, -1))[0][0]
    low = scaler.inverse_transform(lower_bound.reshape(1, -1))[0][0]
    # print(high, avg, low)
    # high_diff = high - avg
    # low_diff = avg - low
    # if ((high_diff + low_diff)/2)/avg < 0.75:
    #     combined = low
    # else:
    #     combined = (low + avg) / 2
    combined = 0

    return high, avg, low, combined

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
