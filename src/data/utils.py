import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.bureau_data import process_bureau_data


def _create_column_description_dict():
    HomeCredit_columns_description = pd.read_csv('data/external/HomeCredit_columns_description.csv', encoding="ISO-8859-1")
    HomeCredit_columns_description.head()

    HomeCredit_columns_description.drop('Unnamed: 0', axis=1, inplace=True)

    column_description_dict = (
        HomeCredit_columns_description.groupby('Table')[['Row', 'Description', 'Special']]
        .apply(
            lambda x: x.set_index('Row').to_dict(orient='index')
        ).to_dict()
    )
    return column_description_dict


COLUMN_DESCRIPTION_DICT = _create_column_description_dict()


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
