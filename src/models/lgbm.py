import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import lightgbm

from src.data.bureau_data import process_bureau_data

RANDOM_SEED = 42


def load_datasets(random_seed):

    train = pd.read_csv('data/external/application_train.csv')
    test = pd.read_csv('data/external/application_test.csv')

    train['is_train_orig'] = 1
    test['is_train_orig'] = 0

    combined = train.append(test)

    object_columns = combined.select_dtypes(include=object).columns

    # Do get_dummies on the entire dataset to avoid future errors
    combined = pd.get_dummies(combined, dummy_na=True, columns=object_columns)

    # Remove non-alphanumeric characters in column names (otherwise LGBM errors)
    combined.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in combined.columns]

    # Add bureau data
    bureau_processed = process_bureau_data()
    combined = combined.merge(bureau_processed, on='SK_ID_CURR', how='left')

    # Split data
    train = combined.copy()[combined['is_train_orig'] == 1]
    train.drop(['is_train_orig'], axis=1, inplace=True)

    test_kaggle = combined.copy()[combined['is_train_orig'] == 0]
    test_kaggle.drop(['is_train_orig'], axis=1, inplace=True)

    # 60/20/20:
    train, test = train_test_split(train, test_size=0.2, random_state=random_seed)
    train, dev = train_test_split(train, test_size=0.25, random_state=random_seed)

    return train, dev, test, test_kaggle


def fit_lgbm():
    train, dev, test, test_kaggle = load_datasets(random_seed=RANDOM_SEED)

    X_train = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_train = train['TARGET']
    X_dev = dev.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_dev = dev['TARGET']

    PARAM_DISTRIBUTIONS = {
        'num_leaves': list(range(20, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
        'subsample_for_bin': list(range(20000, 300000, 20000)),
        'min_child_samples': list(range(20, 500, 5)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100)),
        'is_unbalance': [True, False]
    }
    """
    Taken from https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search
    """

    lgbm_random = RandomizedSearchCV(
        lightgbm.LGBMClassifier(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=80,
        scoring='roc_auc',
        cv=5,
        verbose=3
    )

    lgbm_random.fit(X_train, y_train)

    return lgbm_random


# training_predictions = lgbm_random.predict_proba(X_train)[:, 1]
# roc_auc_score(y_train, training_predictions)
#
# dev_predictions = lgbm_random.predict_proba(X_dev)[:,1]
# roc_auc_score(y_dev, dev_predictions)
#
#
# # Submit
#
# X_test_kaggle = test_kaggle.drop(['SK_ID_CURR', 'TARGET'], axis=1)
# test_kaggle_predictions = lgbm_random.predict_proba(X_test_kaggle)[:, 1]
# test_kaggle['TARGET'] = test_kaggle_predictions
# test_kaggle[['SK_ID_CURR', 'TARGET']].to_csv('data/processed/random_search_lgbm_submission.csv', index=False)
#
# pickle.dump(lgbm_random, open('lgbm_random.pickle', 'wb'))
