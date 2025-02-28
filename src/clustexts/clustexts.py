from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from tqdm import tqdm


PARAMS = {
    'plot_density': True,
    'plot_k': True,
    'show_examples': True,
    'range': (8, 50),
    'min_size': 0,
    'min_gain': 0.03,
    'vectorizer': {
        'max_features': 35000,
        'max_df': 0.5,
        'min_df': 1,
        'use_idf': True
    },
    'reducer': {
        'n_components': 200,
        'n_iter': 20
    }
}


class Clustexts:

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        params = PARAMS
        params.update(dict(kwargs))
        self.__dict__.update(params)
        self._vectorizer = TfidfVectorizer(**self.vectorizer)
        if self.reducer:
            self._reducer = TruncatedSVD(**self.reducer)
    
    def __str__(self):
        return str(self.__dict__)
    
    def __encode(self, X: Iterable[str]) -> np.ndarray:
        X = self._vectorizer.fit_transform(X)
        print(X.shape)
        if self.reducer:
            X = self._reducer.fit_transform(X)
        else:
            X = np.asarray(X.todense())
        print(type(X), X.shape)
        return X
    
    def __getattr__(self, key: str) -> Any:
        return self.__dict__[key]

    
    def __find_best_k(self, X: np.ndarray) -> Tuple[int, KMeans]:
        ks, inertias = [], []
        prev_inertia = None
    
        min_k, max_k = self.range
        for k in range(min_k, max_k + 1):
            kmeansModel = KMeans(n_clusters=k, random_state=42)
            kmeansModel.fit(X)
            inertia = kmeansModel.inertia_
            inertias.append(inertia)
            ks.append(k)
    
            if prev_inertia is not None:
                improvement = (prev_inertia - inertia) / prev_inertia
                
                # Smallest cluster size
                min_cluster_size = \
                    np.min(np.bincount(kmeansModel.labels_))  
    
                if (
                    improvement < self.min_gain
                    or min_cluster_size == self.min_size
                ):
                    print(f"Stopping early at k={k} due to small "
                          f"improvement ({improvement:.4f}) or "
                          "singleton cluster.")
                    break
    
            prev_inertia = inertia

        if self.plot_k:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(min_k, max(ks) + 1), inertias, 'bx-')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Inertia")
            plt.title("Elbow method for best k")

        best_k = ks[-1]
        best_kmeansModel = KMeans(n_clusters=best_k, random_state=42)
        best_kmeansModel.fit(X)
        
        return best_k, best_kmeansModel


    def encode(self, X: Iterable[str]) -> np.ndarray:
        print(X.shape)
        _X = self.__encode(X)
        print(_X.shape)
        return _X


    def __call__(self, X: Iterable[str], explain=False) -> Iterable[int]:
        print(X.shape)
        _X = self.__encode(X)
        print(_X.shape)
        best_k, best_clustering = self.__find_best_k(_X)
        if self.plot_density:
            self.__plot_density(best_k, best_clustering)
        if self.show_examples:
            self.__show_examples(X, best_k, best_clustering)
        return best_clustering.labels_
    

    def __plot_density(self, best_k: int, best_clustering: KMeans) -> None:
        cluster_sizes = np.bincount(best_clustering.labels_)
    
        plt.subplot(1, 2, 2)
        sns.barplot(
            x=np.arange(1, best_k + 1),
            y=cluster_sizes,
            palette="viridis"
        )
        plt.xlabel("Cluster Number")
        plt.ylabel("Number of Items")
        plt.title("Cluster Size Distribution")
    
        plt.tight_layout()
        plt.show()
    
    
    def __show_examples(
        self,
        X: Iterable[str],
        best_k: int,
        best_clustering: KMeans
    ) -> None:
        for cluster_num in range(best_k):
            samples = np.where(best_clustering.labels_ == cluster_num)[0]
            if len(samples) > 3:
                samples = np.random.choice(samples, 3)
            for sample in samples:
                print(f"{cluster_num + 1}: {X.iloc[sample]}")
    

if __name__ == "__main__":
    params = PARAMS.copy()
    params['reducer'] = dict([])
    eklus = Clustexts(**params)
    print(eklus)