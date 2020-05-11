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

NAME_TYPE_SUITE_VALUE_MAP = {
    'Other_A': 'Other',
    'Other_B': 'Other',
    np.nan: 'Other'
}


def impute_null_categorical_features(X):
    X = X.copy()

    # NAME_TYPE_SUITE
    X['NAME_TYPE_SUITE'].replace(NAME_TYPE_SUITE_VALUE_MAP, inplace=True)

    # OCCUPATION_TYPE
    X.loc[
        (X['NAME_INCOME_TYPE'] == 'Unemployed'),
        'OCCUPATION_TYPE'
    ] = 'Unemployed'

    for name_income_type in ['Commercial associate', 'State servant', 'Working', 'Student']:

        X.loc[
            (X['NAME_INCOME_TYPE'] == name_income_type) & (X['OCCUPATION_TYPE'].isnull()),
            'OCCUPATION_TYPE'
        ] = 'Laborers'

    X.loc[
        (X['NAME_INCOME_TYPE'] == 'Pensioner') & (X['OCCUPATION_TYPE'].isnull()),
        'OCCUPATION_TYPE'
    ] = 'Pensioner'

    # FONDKAPREMONT_MODE
    X.drop('FONDKAPREMONT_MODE', axis=1, inplace=True)

