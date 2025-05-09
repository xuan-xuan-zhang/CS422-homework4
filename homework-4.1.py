import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
import numpy as np

try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    df = pd.read_csv(url, sep='\s+',
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                            'acceleration', 'model_year', 'origin', 'car_name'])

    continuous_features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
    X = df[continuous_features]
    X = X.replace('?', np.nan)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=continuous_features)
    df[continuous_features] = X_imputed_df
    hc = AgglomerativeClustering(n_clusters=3, linkage='average', metric='euclidean')
    labels = hc.fit_predict(X_imputed_df)
    df['cluster'] = labels
    cluster_stats = df.groupby('cluster')[continuous_features].agg(['mean', 'var'])
    print("The mean and variance of each cluster:")
    print(cluster_stats)
    class_stats = df.groupby('origin')[continuous_features].agg(['mean', 'var'])
    print("\nThe mean and variance for each category when using origin as the class label:")
    print(class_stats)

except Exception as e:
    print(f"An error occurred: {e}")
