"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler



EstimatorConfig = t.List[t.Dict[str, t.Any]]


def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        hparams = step.get("params", {})
        estimator = estimator_mapping[name](**hparams)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "random-forest-regressor": RandomForestRegressor,
        "gradient-boost-regressor": GradientBoostingRegressor,
        "lineal_regression": LinearRegression
    }

class AverageCostPerRegion(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        data_base = pd.concat(X, y, axis=1)
        data_base = data_base[['smoker','region','y']]
        data_smoker = data_base[data_base.smoker == 'yes']
        self.means_smoker = data_smoker.groupby('region').mean().to_dict()['y']
        data_non_smoker = data_base[data_base.smoker == 'no']
        self.means_non_smoker = data_non_smoker.groupby('region').mean().to_dict()['y']
        return self
    
    def predict(self, X):
        if X.smoker == 'yes':
            return self.means_smoker[X.region] 
        else:
            return self.means_non_smoker[X.region]