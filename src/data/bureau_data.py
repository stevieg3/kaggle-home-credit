import pandas as pd
import numpy as np


def process_bureau_balance_data():
    """
    The raw bureau balance data provides monthly statuses for loans stored in the bureau database. We create aggregate
    features such that the resulting DataFrame has one bureau loan per row.
    """
    bureau_balance = pd.read_csv('data/external/bureau_balance.csv')

    bureau_balance_temp = pd.get_dummies(bureau_balance, columns=['STATUS'])

    # Total statuses over history of loan
    bureau_balance_temp['MONTH_COUNT'] = 1
    bureau_balance_temp_sum = bureau_balance_temp.groupby('SK_ID_BUREAU').sum().reset_index()
    column_rename_dict = dict(
        zip(
            [x for x in bureau_balance_temp.columns if 'STATUS' in x],
            [f'total_{x}' for x in bureau_balance_temp.columns if 'STATUS' in x]
        )
    )
    bureau_balance_temp_sum.rename(columns=column_rename_dict, inplace=True)
    total_statuses_df = bureau_balance_temp_sum.drop('MONTHS_BALANCE', axis=1)

    # Latest loan status
    most_recent_months_balance = bureau_balance_temp.groupby('SK_ID_BUREAU').max()['MONTHS_BALANCE'].reset_index()
    latest_loan_status = bureau_balance_temp.merge(
        most_recent_months_balance,
        on=['SK_ID_BUREAU', 'MONTHS_BALANCE'],
        how='inner'
    )
    latest_loan_status['age_of_latest_bureau_loan_status'] = -latest_loan_status['MONTHS_BALANCE']
    column_rename_dict = dict(
        zip(
            [x for x in latest_loan_status.columns if 'STATUS' in x],
            [f'latest_{x}' for x in latest_loan_status.columns if 'STATUS' in x]
        )
    )
    latest_loan_status.rename(columns=column_rename_dict, inplace=True)
    latest_loan_status['latest_in_arrears'] = np.where(
        latest_loan_status[
            ['latest_STATUS_2', 'latest_STATUS_3', 'latest_STATUS_4', 'latest_STATUS_5']
        ].sum(axis=1) > 0,
        1,
        0
    )
    latest_loan_status.drop(['MONTHS_BALANCE', 'MONTH_COUNT'], axis=1, inplace=True)
    # Let's define 4 or 5 down as being in default:
    latest_loan_status['latest_in_default'] = np.where(
        (latest_loan_status['latest_STATUS_4'] == 1) | (latest_loan_status['latest_STATUS_5'] == 1),
        1,
        0
    )

    # Merge together:
    bureau_balance_features = total_statuses_df.merge(latest_loan_status, on='SK_ID_BUREAU', how='inner')

    bureau_balance_features.columns = [
        col if col == 'SK_ID_BUREAU' else 'bureau_balance_' + col for col in bureau_balance_features.columns
    ]

    return bureau_balance_features


def process_bureau_data():
    """
    Raw bureau data has one bureau loan per row. Multiple loans in the bureau can belong to the same applicant. We
    aggregate this data such that each row represents a single applicant.
    """

    bureau = pd.read_csv('data/external/bureau.csv')
    bureau['CREDIT_TYPE_CATEGORY'] = np.where(
        bureau['CREDIT_TYPE'].isin(
            ['Consumer credit', 'Credit card', 'Car loan', 'Mortgage', 'Microloan', 'Loan for business development']
        ),
        bureau['CREDIT_TYPE'],
        'Other'
    )
    # Object data
    object_columns = bureau.select_dtypes(include=object).columns
    bureau_object_data = pd.get_dummies(
        bureau[['SK_ID_CURR', 'SK_ID_BUREAU'] + list(object_columns)],
        columns=object_columns
    )
    bureau_object_data = bureau_object_data.groupby('SK_ID_CURR').sum().reset_index()
    bureau_object_data.columns = [col if col == 'SK_ID_CURR' else 'COUNT_' + col for col in bureau_object_data.columns]
    bureau_object_data.drop('COUNT_SK_ID_BUREAU', axis=1, inplace=True)

    # Numeric data
    bureau_balance_features = process_bureau_balance_data()
    bureau_full = bureau.merge(bureau_balance_features, on='SK_ID_BUREAU', how='left')
    bureau_full.drop(object_columns, axis=1, inplace=True)
    bureau_full.drop('SK_ID_BUREAU', axis=1, inplace=True)

    bureau_numeric_data = bureau_full.groupby('SK_ID_CURR').agg([np.mean, np.sum, np.min, np.max, np.std]).reset_index()
    bureau_numeric_data.columns = ["_".join(x) for x in bureau_numeric_data.columns.ravel()]
    bureau_numeric_data.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)

    # Combine object and numeric data
    bureau_processed = bureau_numeric_data.merge(bureau_object_data, on='SK_ID_CURR', how='inner')

    assert bureau_processed.shape[0] == bureau['SK_ID_CURR'].nunique()  # Same applicants as original
    assert bureau_processed['SK_ID_CURR'].nunique() == bureau_processed.shape[0]  # No duplicates

    bureau_processed.columns = [
        col if col == 'SK_ID_CURR' else 'bureau_' + col for col in bureau_processed.columns
    ]

    return bureau_processed
