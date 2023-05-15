import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def downsampling(data: pd.DataFrame, ratio: float = 0.7, type: str = "clusters") -> pd.DataFrame:
    """Downsampling of the majority class in a dataset. The downsampling is done in clusters or randomly. 
    data: pd.DataFrame
        Dataset to downsample
    ratio: float
        Ratio of the majority class to downsample
    type: str
        Type of downsampling. It can be "clusters" or "random"
    """

    df = data.copy()
    # downsample in clusters the majority class
    df_majority = df[df['income_50k'] == 0]
    df_minority = df[df['income_50k'] == 1]

    if type == "clusters":
        # Aplicar el algoritmo de clustering (K-Means)
        kmeans = KMeans(n_clusters=10)  # Ajusta el número de clusters deseado
        cluster_labels = kmeans.fit_predict(df_majority)

        # Obtener el tamaño de cada cluster
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

        # Calcular el tamaño objetivo para cada cluster
        target_sizes = (cluster_sizes * ratio).astype(int)

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
        # Downsample majority class
        df_majority_downsampled = df_majority.sample(
            frac=ratio, random_state=42)

        # Separate majority and minority classes
        df_majority = df[df['income'] == '<=50K']
        df_minority = df[df['income'] == '>50K']

        # Downsample majority class
        df_majority_downsampled = df_majority.sample(
            frac=ratio, random_state=42)

    # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled


def preprocessing(data: pd.DataFrame, imputation: str = "mean") -> pd.DataFrame:
    """Preprocessing of the dataset. It drops the unknown column, the duplicates and the columns with more than 40% of missing values. impuation can be "mode" or "knn" """

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

    df['income_50k'] = np.where(df['income_50k'] == ' - 50000.', 0, 1)

    return df
