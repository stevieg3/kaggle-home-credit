import pandas as pd
import numpy as np

CATEGORICAL_FEATURES_WITH_NULLS = [
    'NAME_TYPE_SUITE',
    'OCCUPATION_TYPE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE'
]
"""
Categorical features in training set which had nulls
"""


def preprocess_application_data(X):
    X = X.copy()
    X = impute_null_categorical_features(X)

    # OHE
    object_columns = X.select_dtypes(include=object).columns
    assert X[object_columns].isnull().sum().sum() == 0, "Nulls found in categorical columns"
    X = pd.get_dummies(X, columns=object_columns)

    return X


def impute_null_categorical_features(X):
    X = X.copy()

    # NAME_TYPE_SUITE
    name_type_suite_value_map = {
        'Other_A': 'Other',
        'Other_B': 'Other',
        np.nan: 'Other'
    }

    X['NAME_TYPE_SUITE'].replace(name_type_suite_value_map, inplace=True)

    # OCCUPATION_TYPE
    X.loc[
        (X['NAME_INCOME_TYPE'] == 'Unemployed'),
        'OCCUPATION_TYPE'
    ] = 'Unemployed'


    X.loc[
        (X['NAME_INCOME_TYPE'] == 'Pensioner') & (X['OCCUPATION_TYPE'].isnull()),
        'OCCUPATION_TYPE'
    ] = 'Pensioner'

    for name_income_type in ['Commercial associate', 'State servant', 'Working', 'Student']:

        X.loc[
            (X['NAME_INCOME_TYPE'] == name_income_type) & (X['OCCUPATION_TYPE'].isnull()),
            'OCCUPATION_TYPE'
        ] = 'Laborers'

    # Fill any other nulls with 'Laborers' (e.g. NAME_INCOME_TYPEs which appear in dev/test but were not in training):
    X['OCCUPATION_TYPE'].fillna('Laborers', inplace=True)

    # FONDKAPREMONT_MODE
    X.drop('FONDKAPREMONT_MODE', axis=1, inplace=True)

    # HOUSETYPE_MODE
    X.loc[X['HOUSETYPE_MODE'] != 'block of flats', 'HOUSETYPE_MODE'] = 'not block of flats'

    # WALLSMATERIAL_MODE
    X['WALLSMATERIAL_MODE'].fillna('Unknown', inplace=True)

    # EMERGENCYSTATE_MODE
    X['EMERGENCYSTATE_MODE'].fillna('Unknown', inplace=True)

    object_columns = X.select_dtypes(include=object).columns
    categorical_features_with_nulls = list(X[object_columns].isnull().sum()[X[object_columns].isnull().sum() > 0].index)
    new_categorical_features_with_nulls = set(categorical_features_with_nulls) - set(CATEGORICAL_FEATURES_WITH_NULLS)

    assert len(new_categorical_features_with_nulls) == 0, \
        f"New categorical features with nulls {new_categorical_features_with_nulls}"

    return X
