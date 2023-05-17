import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def downsampling(
    data: pd.DataFrame,
    target: str = 'income_50k',
    target_freq: float = 0.7,
    type: str = "random"
) -> pd.DataFrame:
    """Downsampling of the majority class in a dataset. The downsampling is done in clusters or randomly.
    data: pd.DataFrame
        Dataset to downsample
    target: str
        Name of the target column to balance (default: 'income_50k')
        Must be a binary column
    target_freq: float
        Frequency of apparition of the majority class to downsample
    type: str
        Type of downsampling. It can be "clusters" or "random"
    """

    df = data.copy()

    majority_class = data[target].value_counts().idxmax()
    df_majority = df[df[target] == majority_class]
    df_minority = df[df[target] != majority_class]

    if type == "clusters":
        # Aplicar el algoritmo de clustering (K-Means)
        kmeans = KMeans(n_clusters=10)  # Ajusta el número de clusters deseado
        cluster_labels = kmeans.fit_predict(df_majority)

        # Obtener el tamaño de cada cluster
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

        # Calcular el tamaño objetivo para cada cluster
        target_sizes = (cluster_sizes * target_freq).astype(int)

        # Reducir o replicar las muestras en cada cluster según el tamaño objetivo
        downsampled_samples = []
        for cluster_idx in range(len(cluster_sizes)):
            cluster_samples = df_majority[cluster_labels == cluster_idx]

            if len(cluster_samples) > target_sizes[cluster_idx]:
                # Downsampling: seleccionar aleatoriamente las muestras para alcanzar el tamaño objetivo
                downsampled_samples.append(cluster_samples.sample(
                    n=target_sizes[cluster_idx], random_state=42))
            else:
                # Upsampling: replicar las muestras existentes para alcanzar el tamaño objetivo
                num_replicas = target_sizes[cluster_idx] - len(cluster_samples)
                replicas = cluster_samples.sample(
                    n=num_replicas, replace=True, random_state=42)
                downsampled_samples.append(
                    pd.concat([cluster_samples, replicas]))

        # Concatenar las muestras downsampling de todos los clusters
        downsampled_data = pd.concat(downsampled_samples)
        df_downsampled = pd.concat([downsampled_data, df_minority])

    elif type == "random":
        df_majority_downsampled = df_majority.sample(
            n=int(target_freq/(1-target_freq) * df_minority.shape[0]), random_state=42)
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled


def preprocessing(
    data: pd.DataFrame,
    imputation: str = "mode",
    remove_duplicates: bool = True,
    scaling: str = None
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
    age_bins = [-1, 18, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    # Change to categorical type
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

    return df
