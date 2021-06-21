"""
In this module we store prepare the dataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName], stage: str ):
    df_orginal = reader()
    clean_functions = get_stage()
    df = clean_functions[stage](df_orginal.drop("y", axis = 1))
    y = df_orginal["y"]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
             _fix_data_frame_cat,
             _fix_data_frame_con

        ]
    )
    df = cleaning_fn(df)
    return df

def clean_dataset_h1(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
             _add_new_features_h1,
             _fix_data_frame_cat,
             _fix_data_frame_con

        ]
    )
    df = cleaning_fn(df)
    return df

def clean_dataset_h2(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
             _add_new_features_h2,
             _fix_data_frame_cat,
             _fix_data_frame_con

        ]
    )
    df = cleaning_fn(df)
    return df

def clean_dataset_h3(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
             _add_new_features_h3,
             _fix_data_frame_cat,
             _fix_data_frame_con

        ]
    )
    df = cleaning_fn(df)
    return df

def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _add_new_features_h1(df):
    df['new_feature_1'] = np.where(df['smoker']== 1, 1, np.where(df['bmi'] >= 25, 1, 0))
    
    return df

def _add_new_features_h2(df):
    df['new_feature_2'] = np.where(df['children'] > 1, 0, np.where(df['bmi'] >= 25, 1, 0))

    return df

def _add_new_features_h3(df):
    df['new_feature_3'] = np.where(df['age'] > 52, 1, np.where(df['bmi'] >= 25, 1, 0))

    return df

def _fix_data_frame_cat(df):
    to_get_dummies_cols = get_categorical_column_names()
    df_cat_dummies = pd.get_dummies(df[to_get_dummies_cols], columns= to_get_dummies_cols)
    continius_cols = get_numeric_column_names()
    
    return pd.concat([df[continius_cols], df_cat_dummies], axis = 1)

def _fix_data_frame_con(df):

    age_bins = np.linspace(df['age'].min(),df['age'].max(),11)
    age_labels =['1','2','3','4','5','6','7','8','9','10']
    df['age_buckets'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
    dummie_age = pd.get_dummies(df['age_buckets'], columns='age_buckets')
    df = df.drop(['age','age_buckets'], axis=1)
    df = pd.concat([df, dummie_age], axis=1)
    return df


def _fix_unhandled_nulls(df):
    df.dropna(inplace=True)
    return df


def get_categorical_column_names() -> t.List[str]:
     return (
         "sex,smoker,region"
        
     ).split(",")


def get_binary_column_names() -> t.List[str]:
    return ("sex,smoker,region").split(",")


def get_numeric_column_names() -> t.List[str]:
    return (
        "age,bmi,children"
    ).split(",")


def get_column_names() -> t.List[str]:
    return (
        "age,sex,bmi,children,smoker,region"
    ).split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
        "sex": ("female","male"),
        "smoker": ("no","yes"),
        "region": ("northeast","southwest","northwest","southeast"),
         }

def get_stage()-> t.Dict[str, t.Sequence[str]]:
    return{
        "h_0" : clean_dataset,
        "h_1" : clean_dataset_h1,
        "h_2" : clean_dataset_h2,
        "h_3" : clean_dataset_h3,
    }