import pandas as pd
import numpy as np
import itertools
import time
import ast
from typing import Literal

from joblib import Parallel, delayed

from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, OneSidedSelection

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score


def expand_dicts(df: pd.DataFrame) -> pd.DataFrame:
    '''Expands all the columns that are dictionaries with
    a new column for each key in the dictionary
    df: pd.DataFrame
        Dataframe to expand

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the expanded columns
    '''
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df = pd.concat([df.drop(col, axis=1), df[col].apply(pd.Series)], axis=1)
            except:
                pass
    return df

def read_results(res_file: str):
    """Read the results of a model and preprocessing parameters search
    res_file: str
        Path to the results file
    
    Returns
    -------
    df: pd.DataFrame
        Dataframe with the results of the search
    """
    df = pd.read_csv(res_file)
    for obj_col in ['prep_param', 'model_param']:
        df[obj_col] = df[obj_col].apply(lambda x: ast.literal_eval(x))
    return df


def get_best_params(res_file: str):
    """Get the best parameters from a results file
    res_file: str
        Path to the results file
    
    Returns
    -------
    prep_param: dict
        Best preprocessing parameters
    model_param: dict
        Best model parameters
    """
    df = read_results(res_file)
    df.sort_values(by=['f1_macro'], inplace=True, ascending=False)
    return df.iloc[0]['prep_param'], df.iloc[0]['model_param']


def downsampling(
    data: pd.DataFrame,
    method: Literal[
        'random',
        'NearMiss'
    ] = 'random',
    target: str = 'income_50k',
    target_freq: float = 0.7,
) -> pd.DataFrame:
    """Downsampling of the majority class in a dataset. It uses the RandomUnderSampler or the NearMiss algorithm.
    data: pd.DataFrame
        Dataset to downsample
    method: str
        Method to use for downsampling. It can be "random" or "NearMiss"
    target: str
        Name of the target column
    target_freq: float
        Frequency of the target class after downsampling

    Returns
    -------
    resampled_data: pd.DataFrame
        Resampled dataset
    """
    X = data.drop(target, axis=1)
    y = data[target]

    samples_per_class = {
        0: min(int(target_freq/(1-target_freq) * len(y[y == 1])), len(y[y == 0])),
        1: len(y[y == 1])}
    if method == 'random':
        sampler = RandomUnderSampler(sampling_strategy=samples_per_class)
    elif method == 'NearMiss':
        sampler = NearMiss(sampling_strategy=samples_per_class)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target])], axis=1)

    return resampled_data



def preprocessing(
    data: pd.DataFrame,
    imputation: str = "mode",
    remove_duplicates: bool = True,
    scaling: str = None,
    cat_age: bool = True,
    merge_capital: bool = True,
    generate_dummies: bool = True,
    target: str = 'income_50k',
    downsampling_method: Literal[
        'random',
        'NearMiss'
    ] = 'random',
    target_freq: float | None = None
) -> pd.DataFrame:
    """Preprocessing of the dataset. It removes the unknown column, converts the categorical columns to the categorical type,
    imputes the missing values, scales the numerical columns, generates the dummies for the categorical columns and downsamples
    the majority class.
    data: pd.DataFrame
        Dataset to preprocess
    imputation: str
        Method to use for imputation. It can be "mode" or "knn"
    remove_duplicates: bool
        Whether to remove duplicates or not
    scaling: str
        Method to use for scaling. It can be "minmax" or "standard"
    cat_age: bool
        Whether to convert the age column to categorical or not
    merge_capital: bool
        Whether to merge the capital_gains and capital_losses columns into a single column or not
    generate_dummies: bool
        Whether to generate the dummies for the categorical columns or not
    target: str
        Name of the target column
    downsampling_method: str
        Method to use for downsampling. It can be "random" or "NearMiss"
    target_freq: float
        Frequency of the target class after downsampling

    Returns
    -------
    df: pd.DataFrame
        Preprocessed dataset
    """
    df = data.copy()
    df = df.drop('unknown', axis=1)
    if remove_duplicates:
        df = df.drop_duplicates()

    ########## CATEGORICAL CONVERSION ##########
    for col in ['det_ind_code', 'det_occ_code', 'own_or_self', 'vet_benefits', 'year']:
        df[col] = df[col].astype('category')  # Change to categorical type
    if cat_age:
        age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 85, 95, 105]
        df['age'] = pd.cut(df['age'], bins=age_bins,
                           labels=age_bins[:-1], include_lowest=True).astype('category')
    if merge_capital:
        df['capital_balance'] = df['capital_gains'] - df['capital_losses']
        df = df.drop(['capital_gains', 'capital_losses'], axis=1)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    #############################################

    ########## MISSING VALUES ##########
    cols_with_missing = df.columns[df.isnull().any()]
    df = df.drop(cols_with_missing[df[cols_with_missing].isnull().mean() > 0.4], axis=1)

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
        df[num_col] = (df[num_col] - df[num_col].min()) / (df[num_col].max() - df[num_col].min())
    elif scaling == 'standard':
        df[num_col] = (df[num_col] - df[num_col].mean()) / df[num_col].std()
    ############################

    df['income_50k'] = np.where(df['income_50k'] == ' - 50000.', 0, 1)

    if generate_dummies:
        df = pd.get_dummies(df)

    if target_freq is not None and target_freq < 1:
        df = downsampling(df, method=downsampling_method, target=target, target_freq=target_freq)

    return df


def cross_validation(
    model: object,
    df_tr: pd.DataFrame,
    par_tr: dict,
    par_te: dict | None = None,
    target: str = 'income_50k',
    cv: int = 4,
    mean_score: bool = True,
    n_jobs: int = -1,
    return_predict: bool = False,
    random_state: int  = 42
) -> dict | tuple[dict, np.ndarray, np.ndarray]:
    """Cross validation of a model. It preprocesses the data, fits the model and returns the scores.
    model: object
        Model to fit
    df_tr: pd.DataFrame
        Training dataset
    par_tr: dict
        Parameters for the preprocessing
    par_te: dict
        Parameters for the preprocessing of the test set
    target: str
        Name of the target column
    cv: int
        Number of folds for the cross validation
    mean_score: bool
        Whether to return the mean of the scores or not
    n_jobs: int
        Number of jobs to run in parallel
    return_predict: bool
        Whether to return the predictions or not
    random_state: int
        Random state for the cross validation

    Returns
    -------
    scores: dict
        Dictionary with the scores: accuracy, f1_macro, precision_macro, recall_macro
    y_pred: np.ndarray
        Predictions
    y_true: np.ndarray
        True values
    """
    t1 = time.time()
    model = clone(model)  # to reset the model
    if par_te is None:
        par_te = par_tr.copy()
        par_te['remove_duplicates'] = False
        par_te['target_freq'] = None
        if par_te['imputation'] == 'dropna':
            par_te['imputation'] = 'mode'

    scores = {'accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []}
    
    def cv_fold(train_index, test_index, df_tr):

        df_tr_fold = df_tr.iloc[train_index]
        df_te_fold = df_tr.iloc[test_index]

        df_tr_fold = preprocessing(df_tr_fold, **par_tr)
        df_te_fold = preprocessing(df_te_fold, **par_te)
        df_tr_fold, df_te_fold = df_tr_fold.align(df_te_fold, join='left', axis=1, fill_value=0)

        X_tr, y_tr = df_tr_fold.drop(target, axis=1), df_tr_fold[target]
        X_te, y_te = df_te_fold.drop(target, axis=1), df_te_fold[target]

        print(X_tr.shape, y_tr.shape)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        scr = {}
        scr['accuracy'] = accuracy_score(y_te, y_pred)
        scr['f1_macro'] = f1_score(y_te, y_pred, average='macro')
        scr['precision_macro'] = precision_score(y_te, y_pred, average='macro')
        scr['recall_macro'] = recall_score(y_te, y_pred, average='macro')

        return scr, y_te, y_pred

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)    
    rescv = Parallel(n_jobs=n_jobs)(delayed(cv_fold)(train_index, test_index, df_tr) for train_index, test_index in kf.split(df_tr))

    scores = [x[0] for x in rescv]
    if return_predict:
        y_te = [x[1] for x in rescv]
        y_pr = [x[2] for x in rescv]
        y_te = np.concatenate(y_te, axis=0)
        y_pr = np.concatenate(y_pr, axis=0)

    scores = {k: [d[k] for d in scores] for k in scores[0]}
    if mean_score:
        for score in scores.keys():
            scores[score] = np.mean(scores[score])
    scores['tex'] = time.time() - t1
    return scores if not return_predict else (scores, y_te, y_pr)


def test_preprocess_params(
    model: object,
    prep_grid: dict,
    df: pd.DataFrame,
    cv: int = 4,
    verbose: int = 1,
    ignore_errors: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """Test different preprocessing parameters for a model.
    model: object
        Model to fit
    prep_grid: dict
        Dictionary with the parameters to test
    df: pd.DataFrame
        Training dataset
    cv: int
        Number of folds for the cross validation
    verbose: int
        Verbosity level
    ignore_errors: bool
        Whether to ignore errors or not
    random_state: int
        Random state for the cross validation

    Returns
    -------
    results: pd.DataFrame
        Dataframe with the resulting scores
    """
    c_names = ['prep_param','accuracy','f1_macro','precision_macro','recall_macro','tex']
    results = pd.DataFrame(columns=c_names, dtype=object)

    if not isinstance(prep_grid, list):
        aux = list(itertools.product(*prep_grid.values()))
        prep_grid = [{k: v for k, v in zip(prep_grid.keys(), combination)} for combination in aux]
    
    for i, par_tr in enumerate(prep_grid):
        if verbose == 0: print(f"it: {i+1}/{len(prep_grid)}", end='\r')
        if verbose > 0: print(f"it {i}: Adjusting for {list(par_tr.values())}")

        try:
            par_tr['remove_duplicates'] = True

            cross_val_results = cross_validation(model, df, par_tr, cv=cv, mean_score=True, random_state=random_state)
            results = pd.concat([results, pd.DataFrame({'prep_param': [par_tr]} | cross_val_results)])
        except Exception as e:
            if verbose > 0: print(f"Error in {par_tr}")
            if verbose > 1: print(e)
            if not ignore_errors: raise e

    if verbose == 0: print()
    return results


def test_model_params(
    model: object,
    mod_grid: dict | list,
    df: pd.DataFrame,
    par_tr: dict,
    par_te: dict | None = None,
    cv: int = 4,
    verbose: int = 1,
    ignore_errors: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """Test different model parameters for a model.
    model: object
        Model to fit
    mod_grid: dict | list
        Dictionary with the parameters to test
    df: pd.DataFrame
        Training dataset
    par_tr: dict
        Dictionary with the preprocessing parameters
    par_te: dict | None
        Dictionary with the preprocessing parameters for the test set
    cv: int
        Number of folds for the cross validation
    verbose: int
        Verbosity level
    ignore_errors: bool
        Whether to ignore errors or not
    random_state: int
        Random state for the cross validation

    Returns
    -------
    results: pd.DataFrame
        Dataframe with the resulting scores
    """
    c_names = ['model_param','accuracy','f1_macro','precision_macro','recall_macro','tex']
    results = pd.DataFrame(columns=c_names, dtype=object)

    if not isinstance(mod_grid, list):
        aux = list(itertools.product(*mod_grid.values()))
        mod_grid = [{k: v for k, v in zip(mod_grid.keys(), combination)} for combination in aux]
    for i, par_model in enumerate(mod_grid):
        if verbose == 0: print(f"it: {i+1}/{len(mod_grid)}", end='\r')
        if verbose > 0: print(f"Adjusting for {list(par_model.values())}")
        try:
            model.set_params(**par_model)
            cross_val_results = cross_validation(model, df, par_tr, par_te, cv=cv, mean_score=True, random_state=random_state)
            results = pd.concat([results, pd.DataFrame({'model_param': [par_model]} | cross_val_results)])
        except Exception as e:
            if verbose > 0: print(f"Error in {par_model}")
            if verbose > 1: print(e)
            if not ignore_errors: raise e
    if verbose == 0: print()
    return results


def search_best_combination(
    model: object,
    model_params_grid: dict,
    prep_params_grid: dict,
    df: pd.DataFrame,
    target_metric: Literal['accuracy', 'f1_macro',
                           'precision_macro', 'recall_macro'] = 'f1_macro',
    cv: int = 4,
    N: int = 10,
    verbose: int = 2,
    max_iter: int = 10,
    ignore_errors: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """Search the best combination of preprocessing and model parameters.
    model: object
        Model to fit
    model_params_grid: dict
        Dictionary with the model parameters to test
    prep_params_grid: dict
        Dictionary with the preprocessing parameters to test
    df: pd.DataFrame
        Training dataset
    target_metric: Literal['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        Metric to optimize
    cv: int
        Number of folds for the cross validation
    N: int
        Number of best combinations to keep
    verbose: int
        Verbosity level
    max_iter: int
        Maximum number of iterations
    ignore_errors: bool
        Whether to ignore errors or not

    Returns
    -------
    results: pd.DataFrame
        Dataframe with the resulting scores
    """

    best_mod_param = [{k: v[0] for k, v in model_params_grid.items()}] # list of N best parameters dictionaries
    best_prep_param = []  # list of N best preprocessing dictionaries                                    

    results = pd.DataFrame(columns=['prep_param', 'model_param', 'accuracy',
                            'f1_macro', 'precision_macro', 'recall_macro', 'tex'],
                           index=[0, 1], dtype=object)

    def split_computed(mod_par: list, prep_par: list):
        '''returns a list of all the non computed combinations of preprocessing and model parameters
        and returns a list of the computed model and preprocessing parameters and the results of the
        computed combinations'''
        nonlocal results
        nc_prep_par, nc_mod_par = [], []
        computed = pd.DataFrame(columns=results.columns)
        if isinstance(mod_par, dict) or isinstance(prep_par, dict):
            return mod_par, prep_par, computed
        
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
        prep_par = test_preprocess_params(model, prep_par_list, df, cv=cv,
                                          verbose=verbose-2, ignore_errors=ignore_errors,
                                          random_state=random_state)
        prep_par = prep_par.sort_values(by=target_metric, ascending=False).reset_index(drop=True)
        prep_par['model_param'] = pd.Series([mod_param]*len(prep_par))
        results = pd.concat([results, prep_par])
        best_prep_param = pd.concat([prep_par, computed])['prep_param'][:N].tolist()

    def update_mod_params(prep_param: dict, mod_par_list: list):
        '''searches the best model parameters for a given preprocessing parameters'''''
        nonlocal best_mod_param, results
        mod_par_list, _, computed = split_computed(mod_par_list, [prep_param]*len(mod_par_list))
        mod_par = test_model_params(model, mod_par_list, df, prep_param, cv=cv,
                                    verbose=verbose - 2, ignore_errors=ignore_errors,
                                    random_state=random_state)
        mod_par = mod_par.sort_values(by=target_metric, ascending=False).reset_index(drop=True)
        mod_par['prep_param'] = pd.Series([prep_param]*len(mod_par))
        results = pd.concat([results, mod_par])
        best_mod_param = pd.concat([mod_par, computed])['model_param'][:N].tolist()

    best_metric = 0
    for i in range(1, max_iter+1):
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
