import logging

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def lgbm_feature_reduction(
        initial_fitted_lgbm, X_train, y_train, X_dev, y_dev, num_rounds, lgbm_params=None, prop_to_drop_each_round=0.2
):
    """
    Function starts with an initial LGBM model and finds the ranking of feature importances. The bottom 20% are dropped
    and a new LGBM model is trained. The process is repeated for a specified number of rounds and the train and dev
    ROC AUCs are recorded as well as the models at each round.

    :param initial_fitted_lgbm: Fitted LGBM
    :param X_train: X_train
    :param y_train: y_train
    :param X_dev: X_dev
    :param y_dev: y_dev
    :param num_rounds: Number of rounds of feature dropping to perform
    :param lgbm_params: Parameters to use for new LGBMs. If None default parameters are used
    :param prop_to_drop_each_round: Quantile of features to drop each round
    :return: Dictionary of feature dropping outputs
    """

    X_train = X_train.copy()
    y_train = y_train.copy()
    X_dev = X_dev.copy()
    y_dev = y_dev.copy()

    train_rocauc = roc_auc_score(y_train, initial_fitted_lgbm.predict_proba(X_train)[:, 1])
    dev_rocauc = roc_auc_score(y_dev, initial_fitted_lgbm.predict_proba(X_dev)[:, 1])

    logging.info(f"Initial model: Training ROC AUC: {train_rocauc}, Dev ROC AUC: {dev_rocauc}")

    models = {}
    train_scores = [train_rocauc]
    dev_scores = [dev_rocauc]
    features_after_round = {}

    lgbm_new = initial_fitted_lgbm
    for drop_round in range(1, num_rounds + 1):

        logging.info(f"Round {drop_round} of feature dropping:\n")

        feature_importance_df = get_feature_importance_df(lgbm_new, X_train)

        cut_off = np.quantile(feature_importance_df['importance'], prop_to_drop_each_round)
        features_to_drop = list(feature_importance_df[feature_importance_df['importance'] <= cut_off]['feature'])
        logging.info(f"Number of features with importance below cutoff: {len(features_to_drop)}")

        X_train.drop(columns=features_to_drop, inplace=True)
        X_dev.drop(columns=features_to_drop, inplace=True)

        if lgbm_params:
            lgbm_new = lightgbm.LGBMClassifier(**lgbm_params)
        else:
            lgbm_new = lightgbm.LGBMClassifier()

        lgbm_new.fit(X_train, y_train)

        train_rocauc = roc_auc_score(y_train, lgbm_new.predict_proba(X_train)[:, 1])
        dev_rocauc = roc_auc_score(y_dev, lgbm_new.predict_proba(X_dev)[:, 1])

        logging.info(
            f"After {drop_round} of dropping features: Training ROC AUC: {train_rocauc}, Dev ROC AUC: {dev_rocauc}\n"
        )

        models[drop_round] = lgbm_new
        train_scores.append(train_rocauc)
        dev_scores.append(dev_rocauc)
        features_after_round[drop_round] = list(X_train.columns)

    output = {
        'models': models,
        'train_scores': train_scores,
        'dev_scores': dev_scores,
        'features_after_round': features_after_round
    }

    return output


def get_feature_importance_df(lgbm_model, X):
    """
    Get DataFrame of feature importances for a fitted LGBM model.

    :param lgbm_model: Fitted LGBM model
    :param X: Data trained on
    :return: DataFrame of feature importances
    """
    feature_importance_df = pd.DataFrame(
        dict(
            zip(
                X.columns,
                lgbm_model.feature_importances_
            )
        ),
        index=[0]
    ).T.reset_index().rename(columns={'index': 'feature', 0: 'importance'})

    feature_importance_df.sort_values('importance', ascending=False, inplace=True)

    return feature_importance_df
