import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def downsampling(
    data: pd.DataFrame,
    target: str = 'income_50k',
    target_freq: float = 0.7
) -> pd.DataFrame:
    """Downsampling of the majority class in a dataset. The downsampling is done in clusters or randomly.
    data: pd.DataFrame
        Dataset to downsample
    target: str
        Name of the target column to balance (default: 'income_50k')
    target_freq: float
        Frequency of apparition of the majority class to downsample
    """

    df = data.copy()

    majority_class = data[target].value_counts().idxmax()
    df_majority = df[df[target] == majority_class]
    df_minority = df[df[target] != majority_class]

    if target_freq > df_majority.shape[0]/df.shape[0]:
        return df

    df_majority_downsampled = df_majority.sample(
        n=int(target_freq/(1-target_freq) * df_minority.shape[0]), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled


def preprocessing(
    data: pd.DataFrame,
    imputation: str = "mode",
    remove_duplicates: bool = True,
    scaling: str = None,
    cat_age: bool = True,
    generate_dummies: bool = True
) -> pd.DataFrame:
    """Preprocessing of the dataset. It removes the unknown values, the columns with more than 40% of missing values,
    imputes the missing values with the mode or the KNN algorithm, converts the categorical variables to numerical
    and scales the numerical variables.
    data: pd.DataFrame
        Dataset to preprocess
    imputation: str
        Type of imputation. It can be "mode" or "knn"
    remove_duplicates: bool
        If True, it removes the duplicates
    scaling: str
        Type of scaling. It can be "minmax" or "standard" or None
    """
    df = data.copy()
    df = df.drop('unknown', axis=1)
    if remove_duplicates:
        df = df.drop_duplicates()

    ########## CATEGORICAL CONVERSION ##########
    for col in ['det_ind_code', 'det_occ_code', 'own_or_self', 'vet_benefits', 'year']:
        df[col] = df[col].astype('category')  # Change to categorical type
    if cat_age:
        age_bins = [-1, 18, 25, 35, 45, 55, 65, 75, 85, 95, 105]
        df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_bins[:-1])

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    #############################################

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
    elif imputation == 'drop':
        df = df.dropna()
    elif imputation == 'nancat':
        for col in cols_with_missing:
            df[col] = df[col].cat.add_categories("Missing")
            df[col] = df[col].fillna("Missing")

    ###################################

    ########## SCALING ##########
    num_col = df.select_dtypes(include=['int64', 'float64']).columns
    if scaling == 'minmax':
        df[num_col] = (df[num_col] - df[num_col].min()) / \
            (df[num_col].max() - df[num_col].min())
    elif scaling == 'standard':
        df[num_col] = (df[num_col] - df[num_col].mean()) / \
            df[num_col].std()
    ############################

    df['income_50k'] = np.where(df['income_50k'] == ' - 50000.', 0, 1)

    if generate_dummies:
        df = pd.get_dummies(df)

    return df
