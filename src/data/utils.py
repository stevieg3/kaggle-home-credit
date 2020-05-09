import pandas as pd


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
