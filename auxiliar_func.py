import pandas as pd
import numpy as np


def preprocessing(data: pd.DataFrame, imputation: str = "mean") -> pd.DataFrame:
    df = data.copy()
    df = df.drop('unknown', axis=1)  # Drop unknown column
    df = df.drop_duplicates()  # Drop duplicates

    ########## MISSING VALUES ##########
    # Get columns with missing values
    cols_with_missing = df.columns[df.isnull().any()]
    # Drop columns with more than 40% of missing values
    df = df.drop(
        cols_with_missing[df[cols_with_missing].isnull().mean() > 0.4], axis=1)

    cols_with_missing = df.columns[df.isnull().any()]
    if imputation == 'mode':
        for col in cols_with_missing:
            df[col] = df[col].fillna(df[col].value_counts().index[0])
    elif imputation == 'knn':
        pass
    ###################################

    for col in ['det_ind_code', 'det_occ_code', 'own_or_self', 'vet_benefits', 'year']:
        df[col] = df[col].astype('category')  # Change to categorical type

    age_bins = [-1, 18, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    # Change to categorical type
    df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_bins[:-1])

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    return df
