from functools import reduce

import pandas as pd
import numpy as np


def process_all_other_data():
    previous_application = process_previous_application()
    assert previous_application['SK_ID_CURR'].nunique() == previous_application.shape[0]
    print('Finished previous_application!')

    installments_payments = process_installments_payments()
    assert installments_payments['SK_ID_CURR'].nunique() == installments_payments.shape[0]
    print('Finished installments_payments!')

    pos_cash_balance = process_pos_cash_balance()
    assert pos_cash_balance['SK_ID_CURR'].nunique() == pos_cash_balance.shape[0]
    print('Finished pos_cash_balance!')

    credit_card_balance = process_credit_card_balance()
    assert credit_card_balance['SK_ID_CURR'].nunique() == credit_card_balance.shape[0]
    print('Finished credit_card_balance!')

    dfs = [previous_application, installments_payments, pos_cash_balance, credit_card_balance]
    df_final = reduce(lambda left, right: pd.merge(left, right, on='SK_ID_CURR', how='outer'), dfs)
    assert df_final['SK_ID_CURR'].nunique() == df_final.shape[0]

    return df_final


def process_previous_application():
    previous_application = pd.read_csv('data/external/previous_application.csv')
    previous_application_orig = previous_application.copy()

    # aggregation 1
    previous_application.drop('SK_ID_PREV', axis=1, inplace=True)
    previous_application_object_data = _aggregate_object_columns(previous_application, 'SK_ID_CURR')

    # Numeric data
    previous_application_numeric_data = previous_application_orig.copy()
    previous_application_numeric_data.drop(['SK_ID_PREV'], axis=1, inplace=True)

    # aggregation 1
    previous_application_numeric_data = _aggregate_numeric_columns(previous_application_numeric_data, 'SK_ID_CURR')

    previous_application_processed = previous_application_numeric_data.merge(
        previous_application_object_data, on='SK_ID_CURR'
    )

    previous_application_processed.columns = [
        col if col == 'SK_ID_CURR' else 'POS_CASH_balance_' + col for col in previous_application_processed.columns
    ]

    return previous_application_processed


def process_installments_payments():
    installments_payments = pd.read_csv('data/external/installments_payments.csv')

    id_df = installments_payments.groupby(['SK_ID_PREV', 'SK_ID_CURR']).count().reset_index()[
        ['SK_ID_PREV', 'SK_ID_CURR']
    ]
    id_df['comb_id'] = id_df.index + 1
    installments_payments = installments_payments.merge(id_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    installments_payments.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    # aggregation 1
    installments_payments_numeric_data = _aggregate_numeric_columns(installments_payments, 'comb_id')

    installments_payments_numeric_data = installments_payments_numeric_data.merge(id_df, on=['comb_id'])
    installments_payments_numeric_data.drop(['SK_ID_PREV', 'comb_id'], axis=1, inplace=True)

    # aggregation 2
    installments_payments_numeric_data = _aggregate_numeric_columns(installments_payments_numeric_data, 'SK_ID_CURR')

    installments_payments_numeric_data.columns = [
        col if col == 'SK_ID_CURR' else 'POS_CASH_balance_' + col for col in installments_payments_numeric_data.columns
    ]

    return installments_payments_numeric_data


def process_pos_cash_balance():
    # POS_CASH_balance - need to do 2 levels of aggregation
    pos_cash_balance = pd.read_csv('data/external/POS_CASH_balance.csv')
    pos_cash_balance_orig = pos_cash_balance.copy()

    # Object data
    id_df = pos_cash_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR']).count().reset_index()[['SK_ID_PREV', 'SK_ID_CURR']]
    id_df['comb_id'] = id_df.index + 1
    pos_cash_balance = pos_cash_balance.merge(id_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    pos_cash_balance.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    # aggregation 1
    pos_cash_balance_object_data = _aggregate_object_columns(pos_cash_balance, 'comb_id')

    pos_cash_balance_object_data = pos_cash_balance_object_data.merge(id_df, on=['comb_id'])
    pos_cash_balance_object_data.drop(['SK_ID_PREV', 'comb_id'], axis=1, inplace=True)
    # aggregation 2
    pos_cash_balance_object_data = pos_cash_balance_object_data.groupby('SK_ID_CURR').sum().reset_index()
    pos_cash_balance_object_data.columns = [
        col if col == 'SK_ID_CURR' else 'COUNT_' + col for col in pos_cash_balance_object_data.columns
    ]

    # Numeric data
    pos_cash_balance_numeric_data = pos_cash_balance_orig.copy()
    pos_cash_balance_numeric_data = pos_cash_balance_numeric_data.merge(id_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    pos_cash_balance_numeric_data.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    # aggregation 1
    pos_cash_balance_numeric_data = _aggregate_numeric_columns(pos_cash_balance_numeric_data, 'comb_id')

    pos_cash_balance_numeric_data = pos_cash_balance_numeric_data.merge(id_df, on=['comb_id'])
    pos_cash_balance_numeric_data.drop(['SK_ID_PREV', 'comb_id'], axis=1, inplace=True)

    # aggregation 2
    pos_cash_balance_numeric_data = _aggregate_numeric_columns(pos_cash_balance_numeric_data, 'SK_ID_CURR')

    pos_cash_balance_processed = pos_cash_balance_numeric_data.merge(pos_cash_balance_object_data, on='SK_ID_CURR')

    pos_cash_balance_processed.columns = [
        col if col == 'SK_ID_CURR' else 'POS_CASH_balance_' + col for col in pos_cash_balance_processed.columns
    ]

    return pos_cash_balance_processed


def process_credit_card_balance():
    # credit_card_balance - need to do 2 levels of aggregation
    credit_card_balance = pd.read_csv('data/external/credit_card_balance.csv')
    credit_card_balance_orig = credit_card_balance.copy()

    # Object data
    id_df = credit_card_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR']).count().reset_index()[['SK_ID_PREV', 'SK_ID_CURR']]
    id_df['comb_id'] = id_df.index + 1
    credit_card_balance = credit_card_balance.merge(id_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    credit_card_balance.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    # aggregation 1
    credit_card_balance_object_data = _aggregate_object_columns(credit_card_balance, 'comb_id')

    credit_card_balance_object_data = credit_card_balance_object_data.merge(id_df, on=['comb_id'])
    credit_card_balance_object_data.drop(['SK_ID_PREV', 'comb_id'], axis=1, inplace=True)
    # aggregation 2
    credit_card_balance_object_data = credit_card_balance_object_data.groupby('SK_ID_CURR').sum().reset_index()
    credit_card_balance_object_data.columns = [
        col if col == 'SK_ID_CURR' else 'COUNT_' + col for col in credit_card_balance_object_data.columns
    ]

    # Numeric data
    credit_card_balance_numeric_data = credit_card_balance_orig.copy()
    credit_card_balance_numeric_data = credit_card_balance_numeric_data.merge(id_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    credit_card_balance_numeric_data.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    # aggregation 1
    credit_card_balance_numeric_data = _aggregate_numeric_columns(credit_card_balance_numeric_data, 'comb_id')

    credit_card_balance_numeric_data = credit_card_balance_numeric_data.merge(id_df, on=['comb_id'])
    credit_card_balance_numeric_data.drop(['SK_ID_PREV', 'comb_id'], axis=1, inplace=True)

    # aggregation 2
    credit_card_balance_numeric_data = _aggregate_numeric_columns(credit_card_balance_numeric_data, 'SK_ID_CURR')

    credit_card_balance_processed = credit_card_balance_numeric_data.merge(credit_card_balance_object_data, on='SK_ID_CURR')

    credit_card_balance_processed.columns = [
        col if col == 'SK_ID_CURR' else 'POS_CASH_balance_' + col for col in credit_card_balance_processed.columns
    ]

    return credit_card_balance_processed


def _aggregate_object_columns(df, uid_column):
    object_columns = df.select_dtypes(include=object).columns
    object_data = pd.get_dummies(
        df[[uid_column] + list(object_columns)],
        columns=object_columns
    )
    object_data = object_data.groupby(uid_column).sum().reset_index()
    object_data.columns = [col if col == uid_column else 'COUNT_' + col for col in object_data.columns]

    return object_data


def _aggregate_numeric_columns(df, uid_column):
    numeric_columns = df.select_dtypes(include=np.number).columns
    numeric_data = df.copy()[numeric_columns]

    numeric_data = numeric_data.groupby(uid_column).agg([np.mean, np.sum, np.min, np.max]).reset_index()

    numeric_data.columns = ["_".join(x) for x in numeric_data.columns.ravel()]
    numeric_data.rename(columns={f'{uid_column}_': f'{uid_column}'}, inplace=True)

    return numeric_data
