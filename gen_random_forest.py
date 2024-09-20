import sys
import asyncio
import json
import numpy as np
import pandas as pd
import ast
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from gen_embeddings import get_all_embeddings
from datetime import datetime
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler
import re
from sklearn.compose import ColumnTransformer
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import cProfile
import pstats
from functools import lru_cache
from openai import AsyncOpenAI

client = AsyncOpenAI()

def total_processor(data):
    scaler = RobustScaler()
    data['price'] = pd.to_numeric(data['price'].str.replace('[$,]', '', regex=True), errors='coerce')
    data['shipping'] = pd.to_numeric(data['shipping'].str.replace('[$,]', '', regex=True), errors='coerce')
    data['quantity'] = data['quantity'].clip(lower=1)  # Ensure no zero quantities
    totals = (data['price'] + data['shipping']) / data['quantity']
    totals_reshaped = totals.values.reshape(-1, 1)
    totals_scaled = scaler.fit_transform(totals_reshaped).ravel()
    return totals_scaled, scaler

async def get_quantities_batch(titles, batch_size=20):
    quantities = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"For each of the following product listings, how many items are included? Do not include peripherals. Respond with just the numbers, separated by commas: {', '.join(batch)}"}],
        )
        batch_quantities = response.choices[0].message.content.strip().split(',')
        quantities.extend(batch_quantities)
    return quantities

class preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_scaler = StandardScaler()
        self.embedding_scaler = StandardScaler()
        self.start_date = pd.Timestamp('2021-01-01')
        start_date = datetime.strptime('2021-01-01', '%Y-%m-%d')
        end_date = datetime.today()
        self.max_days = (end_date - start_date).days
        complete_dates = pd.date_range(start=start_date, end=end_date)
        complete_ordinal_dates = complete_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)
        complete_ordinal_dates_df = pd.DataFrame(complete_ordinal_dates, columns=['ordinal_date'])
        self.date_scaler = RobustScaler()
        self.date_scaler.fit(complete_ordinal_dates_df)
        self._embeddings_dict = {}

    def fit(self, X, y=None, sample_weight=None):
        if 'embeddings' not in X.columns:
            titles = self._process_title(X)
            X['embeddings'] = self._get_embeddings(titles)
        self.embedding_scaler.fit(np.vstack(X['embeddings']))
        return self

    def transform(self, X, sample_weight=None):
        if 'embeddings' not in X.columns:
            titles = self._process_title(X)
            X['embeddings'] = self._get_embeddings(titles)
        embeddings_scaled = self.embedding_scaler.transform(np.vstack(X['embeddings']))
        if 'ordinal_date' not in X.columns:
            X['ordinal_date'] = self._process_date(X)
        scaled_dates = self.date_scaler.transform(X[['ordinal_date']])
        features = np.hstack((embeddings_scaled, scaled_dates))
        return features

    def _process_title(self, X):
        lowered = X['title'].str.lower()
        cleaned = lowered.str.replace(r'[^a-zA-Z0-9_\s]', '', regex=True)
        spaces = cleaned.str.replace(r'\s+', ' ', regex=True)
        return spaces

    def _process_date(self, X):
        return (pd.to_datetime(X['date'], format='%b %d, %Y') - self.start_date).dt.days.to_frame()

    def _get_embeddings(self, titles):
        unique_titles = list(set(titles))
        new_titles = [title for title in unique_titles if title not in self._embeddings_dict]

        if new_titles:
            new_embeddings = get_all_embeddings(new_titles)
            self._embeddings_dict.update(dict(zip(new_titles, new_embeddings)))

        return [self._embeddings_dict[title] for title in titles]

    def _get_quantities(self, titles):
        quantities = []
        unique_titles = list(set(titles))
        loop = asyncio.get_event_loop()
        batch_quantities = loop.run_until_complete(get_quantities_batch(unique_titles))
        quantity_dict = dict(zip(unique_titles, batch_quantities))

        for title in titles:
            quantity = quantity_dict.get(title, '1')
            try:
                quantity = float(quantity)
            except ValueError:
                quantity = 1.0
            quantities.append(quantity)
        return quantities

def get_cv_strategy(n_samples):
    if n_samples < 100:
        return LeaveOneOut()
    else:
        return KFold(n_splits=10, shuffle=True)

def get_random_param_grid(n_samples):
    if n_samples < 50:
        return {
            'regressor__max_depth': [2],
            'regressor__min_samples_leaf': [6, 7, 8, 10],
            'regressor__min_samples_split': [15, 20, 25],
            'regressor__n_estimators': [5, 10, 15]
        }
    else:
        return {
            'regressor__max_depth': [3],
            'regressor__min_samples_leaf': [6, 7, 8, 10],
            'regressor__min_samples_split': [20, 25, 30],
            'regressor__n_estimators': [15, 20, 25]
        }
    
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        percentage_errors = np.abs((y_true - y_pred) / y_true)[non_zero_mask] * 100
        mape = np.mean(percentage_errors)
        return mape
    else:
        return np.inf

def gen_random_forest(data):
    preprocessor_instance = preprocessor()
    titles = preprocessor_instance._process_title(data)
    embeddings = preprocessor_instance._get_embeddings(titles)
    data['embeddings'] = embeddings
    ordinal_dates = preprocessor_instance._process_date(data)
    data['ordinal_date'] = ordinal_dates
    quantities = preprocessor_instance._get_quantities(data['title'])
    data['quantity'] = quantities
    max_days = preprocessor_instance.max_days

    model = Pipeline([
        ('preprocessor', preprocessor()),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    random_param_grid = get_random_param_grid(data.shape[0])
    adaptive_cv = get_cv_strategy(data.shape[0])
    grid_search = GridSearchCV(model, random_param_grid, cv=adaptive_cv, scoring='neg_mean_absolute_error', n_jobs=-1)

    data['sold'] = pd.to_numeric(data['sold'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
    total_sold = data['sold'].sum()
    sold_per_day = total_sold / max_days
    sold_per_seller_per_day = sold_per_day/2

    weights = data['sold']
    data['weights'] = weights

    X = data[['title', 'ordinal_date', 'embeddings']]
    y, scaler = total_processor(data)

    data['total'] = scaler.inverse_transform(y.reshape(-1, 1))
    data.to_csv('testing.csv', index=False)

    print('starting gridsearch')
    grid_search.fit(X, y)

    model = grid_search.best_estimator_

    mape = 0
    gap = 0

    print('training complete')
    return data, model, scaler, mape, gap
