import pandas as pd
import numpy as np
import itertools
from typing import Literal

from sklearn.cluster import KMeans
from sklearn.base import clone
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
    model = clone(model)  # to reset the model
    if par_te is None:
        par_te = par_tr.copy()
        par_te['remove_duplicates'] = False
        par_te['target_freq'] = None
        if par_te['imputation'] == 'dropna':
            par_te['imputation'] = 'mode'

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
    model: object,
    prep_grid: dict,
    df: pd.DataFrame,
    metrics: list = ['accuracy', 'f1_macro',
                     'precision_macro', 'recall_macro'],
    cv: int = 4,
    verbose: int = 1
) -> pd.DataFrame:
    c_names = ['prep_param'] + metrics
    results = pd.DataFrame(columns=c_names, dtype=object)

    if type(prep_grid) != list:
        aux = list(itertools.product(*prep_grid.values()))
        prep_grid = [{k: v for k, v in zip(prep_grid.keys(), combination)} for combination in aux]
    for par_tr in prep_grid:
        if verbose > 0:
            print(f"Adjusting for {list(par_tr.values())}")
        try:
            par_tr['remove_duplicates'] = True

            cross_val_results = cross_validation(model, df, par_tr, cv=cv,
                                                 scoring=metrics)
            results = pd.concat([results, pd.DataFrame(
                {'prep_param': [par_tr]} | cross_val_results)])
        except Exception as e:
            if verbose > 0:
                print(f"Error in {par_tr}")
            if verbose > 1:
                print(e)

    return results


def test_model_params(
    model: object,
    mod_grid: dict | list,
    df: pd.DataFrame,
    par_tr: dict,
    par_te: dict | None = None,
    metrics: list = ['accuracy', 'f1_macro',
                     'precision_macro', 'recall_macro'],
    cv: int = 4,
    verbose: int = 1
) -> pd.DataFrame:
    c_names = ['model_param'] + metrics
    results = pd.DataFrame(columns=c_names, dtype=object)

    if type(mod_grid) != list:
        aux = list(itertools.product(*mod_grid.values()))
        mod_grid = [{k: v for k, v in zip(mod_grid.keys(), combination)} for combination in aux]
    for par_model in mod_grid:
        if verbose > 0:
            print(f"Adjusting for {list(par_model.values())}")
        try:
            model.set_params(**par_model)

            cross_val_results = cross_validation(model, df, par_tr, par_te, cv=cv,
                                                 scoring=metrics)
            results = pd.concat([results, pd.DataFrame(
                {'model_param': [par_model]} | cross_val_results)])
        except Exception as e:
            if verbose > 0:
                print(f"Error in {par_model}")
            if verbose > 1:
                print(e)

    return results

# tests the best preprocessing combination with a model
# keep the best 5 preprocessing combinations
# with the best preprocessing combination, find the best model parameters
# keep the best 5 model parameters
# repeat the last two steps searching over the best 5 preprocessing combinations and 5 model parameters until no improvement is found
# return a dataframe with all the trained models and their metrics sorted by the target metric
def search_best_combination(
    model: object,
    model_params_grid: dict,
    prep_params_grid: dict,
    df: pd.DataFrame,
    target_metric: Literal['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] = 'f1_macro',
    cv: int = 4,
    N: int = 5,
    verbose: int = 1,
    max_iter: int = 10
) -> pd.DataFrame:

    best_mod_param = [{k: v[0] for k, v in model_params_grid.items()}]
    best_prep_param = []

    results = pd.DataFrame(columns=['prep_param', 'model_param',
                           'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
                           index=[0,1], dtype=object)
    
    
    def split_computed(mod_par: list, prep_par: list):
        '''returns a list of all the non computed combinations of preprocessing and model parameters
        and returns a list of the computed model and preprocessing parameters and the results of the
        computed combinations'''
        nonlocal results
        nc_prep_par, nc_mod_par, computed = [], [], pd.DataFrame(columns=results.columns)
        if isinstance(mod_par, dict) or isinstance(prep_par, dict): return mod_par, prep_par, computed
        for m_par, p_par in zip(mod_par, prep_par):
            row = results[(results['prep_param'] == p_par) & (results['model_param'] == m_par)]
            if row.empty:
                nc_prep_par.append(p_par)
                nc_mod_par.append(m_par)
            else:
                computed = pd.concat([computed, row])
        return nc_mod_par, nc_prep_par, computed

    def update_prep_params(mod_param: dict, prep_par_list: list):
        '''searches the best preprocessing parameters for a given model parameters'''''
        nonlocal best_prep_param, results
        model.set_params(**mod_param)
        _, prep_par_list, computed = split_computed([mod_param]*len(prep_par_list), prep_par_list)
        prep_par = test_preprocess_params(model, prep_par_list, df, cv=cv, verbose=verbose-2).sort_values(by=target_metric, ascending=False).reset_index(drop=True)
        prep_par['model_param'] = pd.Series([mod_param]*len(prep_par))
        results = pd.concat([results, prep_par])
        best_prep_param = pd.concat([prep_par,computed])['prep_param'][:N].tolist()

    def update_mod_params(prep_param: dict, mod_par_list: list):
        '''searches the best model parameters for a given preprocessing parameters'''''
        nonlocal best_mod_param, results
        mod_par_list, _, computed = split_computed(mod_par_list, [prep_param]*len(mod_par_list))
        mod_par = test_model_params(model, mod_par_list, df, prep_param, cv=cv, verbose=verbose-2).sort_values(by=target_metric, ascending=False).reset_index(drop=True)
        mod_par['prep_param'] = pd.Series([prep_param]*len(mod_par))
        results = pd.concat([results, mod_par])
        best_mod_param = pd.concat([mod_par,computed])['model_param'][:N].tolist()

    best_metric = 0
    for i in range(1,max_iter+1):
        if verbose > 0: print(f'===Iteration {i}===')
        if verbose > 0: print(f'Searching preprocessing parameters...')
        update_prep_params(best_mod_param[0], prep_params_grid)
        if verbose > 0: print(f'Searching model parameters...')
        update_mod_params(best_prep_param[0], model_params_grid)

        results = results.sort_values(by=target_metric, ascending=False).reset_index(drop=True).dropna()

        if results[target_metric].max() > best_metric:
            best_metric = results[target_metric].max()
        else:
            break
        
        # update the parameter grid without duplicates
        results['temp1'] = results['prep_param'].astype(str)
        prep_params_grid = results.drop_duplicates(subset='temp1').drop('temp1', axis=1)['prep_param'][:N].tolist()
        results['temp2'] = results['model_param'].astype(str)
        model_params_grid = results.drop_duplicates(subset='temp2').drop('temp2', axis=1)['model_param'][:N].tolist()

        results = results.drop(['temp1', 'temp2'], axis=1)

        if verbose > 0:
            print(f"Best metric: {best_metric}")
            if verbose > 1:
                print(f"Best preprocessing parameters: {best_prep_param}")
                print(f"Best model parameters: {best_mod_param}")

    return results.sort_values(by=target_metric, ascending=False).reset_index(drop=True)
