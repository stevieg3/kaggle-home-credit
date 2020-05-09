import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
import lightgbm

train = pd.read_parquet('data/interim/train.parquet')
dev = pd.read_parquet('data/interim/dev.parquet')

prop_nulls_df = (train.isnull().sum() / len(train)).reset_index().rename(
    columns={'index': 'feature', 0: 'proportion_of_nulls'}
)
FEATURES_TO_DROP = list(prop_nulls_df[prop_nulls_df['proportion_of_nulls'] > 0.2]['feature'])


class BaselineModel:
    def __init__(self):
        pass

    def drop_columns(self, X, columns_to_drop):
        return X.drop(columns_to_drop, axis=1)

    def convert_object_columns_to_string(self, X):
        object_columns = X.select_dtypes(include=object).columns
        X[object_columns] = X[object_columns].astype(str)
        return X

    def create_pipeline(self):
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(self.drop_columns, kw_args={'columns_to_drop': FEATURES_TO_DROP})),
            # Need to convert to string otherwise leads to error when imputing object data:
            ('convert_object_columns_to_string', FunctionTransformer(self.convert_object_columns_to_string))
        ])

        # Pipeline of operations to perform on any object columns in DataFrame
        object_pipeline = Pipeline(
            [
                ('most_frequent_imputer', SimpleImputer(strategy='most_frequent')),  # Very slow
                ('ohe', OneHotEncoder())
            ]
        )

        # Pipeline of operations to perform on any numeric columns in DataFrame
        numeric_pipeline = Pipeline(
            [
                ('mean_imputer', SimpleImputer(strategy='mean')),
                ('min_max_scalar', MinMaxScaler())
            ]
        )

        full_pipeline = Pipeline(
            [
                (
                    'process_data',
                    ColumnTransformer(
                        [
                            ('numeric_processing', numeric_pipeline, make_column_selector(dtype_include=np.number)),
                            ('object_processing', object_pipeline, make_column_selector(dtype_include=object))
                        ]
                    )
                )
            ]
        )

        end_to_end_pipeline = Pipeline(
            [
                ('preprocessing', preprocessing_pipeline),
                ('processing', full_pipeline),
                ('model', lightgbm.LGBMClassifier())
            ]
        )

        return end_to_end_pipeline

    def fit_pipeline(self, X, y):
        end_to_end_pipeline = self.create_pipeline()
        end_to_end_pipeline.fit(X, y)
        return end_to_end_pipeline


if __name__ == "__main__":
    X_train = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y_train = train['TARGET']
    baseline_model = BaselineModel()
    end_to_end_pipeline = baseline_model.fit_pipeline(X_train, y_train)
    pickle.dump(end_to_end_pipeline, open('models/baseline_model_pipeline.pickle', 'wb'))
