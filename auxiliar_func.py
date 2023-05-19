import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score


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

    if target_freq >= df_majority.shape[0]/df.shape[0]:
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
    generate_dummies: bool = True,
    target: str = 'income_50k',
    target_freq: float | None = None
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
    elif imputation == 'dropna':
        df = df.dropna()
    elif imputation == 'nacat':
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

    if target_freq is not None and target_freq < 1:
        df = downsampling(df, target, target_freq)

    return df


def cross_validation(
    model: object,
    df_tr: pd.DataFrame,
    par_tr: dict,
    par_te: dict | None = None,
    target: str = 'income_50k',
    cv: int = 4,
    scoring: list = ['accuracy', 'f1_macro',
                     'precision_macro', 'recall_macro'],
    mean_score: bool = True
) -> dict:
    """Cross validation of a model. It returns the mean of the metrics of the cross validation.
    model: object
        Model to cross validate
    df_tr: pd.DataFrame
        Training dataset
    par_tr: dict
        Parameters to preprocess the training dataset
    par_te: dict
        Parameters to preprocess the test dataset
    cv: int
        Number of folds for the cross validation
    scoring: list
        Metrics to calculate in the cross validation
    """
    if par_te is None:
        par_te = par_tr.copy()
        par_te['remove_duplicates'] = False
        par_te['target_freq'] = None

    scores = {}
    for score in scoring:
        scores[score] = []

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(df_tr):
        df_tr_fold = df_tr.iloc[train_index]
        df_te_fold = df_tr.iloc[test_index]

        df_tr_fold = preprocessing(df_tr_fold, **par_tr)
        df_te_fold = preprocessing(df_te_fold, **par_te)
        df_tr_fold, df_te_fold = df_tr_fold.align(
            df_te_fold, join='left', axis=1, fill_value=0)

        X_tr, y_tr = df_tr_fold.drop(target, axis=1), df_tr_fold[target]
        X_te, y_te = df_te_fold.drop(target, axis=1), df_te_fold[target]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        scores['accuracy'].append(accuracy_score(y_te, y_pred))
        scores['f1_macro'].append(f1_score(y_te, y_pred, average='macro'))
        scores['precision_macro'].append(
            precision_score(y_te, y_pred, average='macro'))
        scores['recall_macro'].append(
            recall_score(y_te, y_pred, average='macro'))

    if mean_score:
        for score in scores.keys():
            scores[score] = np.mean(scores[score])

    return scores


def test_preprocess_params(
    df: pd.DataFrame,
    model: object,
    params: dict,
    metrics: list = ['accuracy', 'f1_macro',
                     'precision_macro', 'recall_macro'],
    cv: int = 4,
    verbose: int = 1
) -> pd.DataFrame:
    c_names = list(params.keys()) + metrics
    results = pd.DataFrame(columns=c_names)

    for combination in list(itertools.product(*params.values())):
        if verbose > 0:
            print(f"Adjusting for {combination}")
        par_tr = {k: v for k, v in zip(params.keys(), combination)}
        par_tr['remove_duplicates'] = True

        cross_val_results = cross_validation(model, df, par_tr, cv=cv,
                                             scoring=metrics)
        results = pd.concat([results, pd.DataFrame([list(combination) + list(cross_val_results.values())],
                                                   columns=c_names)])

    return results
