from doctest import testmod
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
    'range': (2, 20),
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
        'n_iter': 20,
        'random_state': None
    },
    'plot_density': False,
    'plot_k': False,
    'show_examples': False,
    'verbose': False
}



class Clustexts:

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        Performs k-means clustering on a collection of texts. It automates the selection of `k` by running the elbow method implicitly. The algorithm only expects the range of minimum and maximum values for `k` (default to 2 and 20, respectively).
        
        Texts are encoded using a TFIDF Bag-of-Words representation. Optionally, Truncated Singular Value Decomposition can be used to reduce the dimensionality of the resulting matrix and project the topology onto an embedded space, for improved data compression and schema generalization.
        
        The call to returns an iterator containing the cluster identifiers associated with each input document.


        Examples
        --------
        >>> rows = [
        ...   'one text',
        ...   'another text',
        ...   'this sentence',
        ...   'fourth sentence',
        ...   'fifth sentence',
        ... ]
        >>> df = pd.DataFrame(rows, columns=['text'])
        
        >>> cls = Clustexts(
        ...   reducer={},
        ...   range = (2, 10),
        ...   min_gain=0.001,
        ...   vectorizer={'min_df': 0.0}
        ... )
        >>> df['cluster'] = cls(df['text'])

        >>> cls = Clustexts(
        ...   min_size = 3,
        ...   min_gain = 0.001,
        ...   range = (2, 10),
        ...   vectorizer = {
        ...     'max_features': 100
        ...   },
        ...   reducer={}
        ... )
        >>> df['cluster'] = cls(df['text'])

        Parameters
        ----------
        
        -- Clustering --
        
            range: Tuple[int, int] = (2, 20)
                Specifies the minimum and maximum values of `k` to explore when applying the elbow method.

            min_size: int = 0
                The minimum cluster size to be accepted. If reached, the clustering stops.
                
            min_gain: float = 0.03
                The minimum relative improvement for the clustering to continue running (as a percentage of the inertia). If the relative improvement becomes smaller than this value at any point, the clustering stops.
        

        -- Vectorization (required) --

            All these parameters are used with their standard meaning in scikit-learn. Refer to the package's documentation.
            ```
            vectorizer: Dict[str, Any]
                max_features: int = 35_000
                max_df: int | float = 0.5
                min_df: int | float = 1
                use_idf: bool = True
                ...
            ```


        -- Reporting (optional) --
        
            plot_density: bool = False
                If set to `True`, the system will plot cluster densities (number of documents in each cluster).
                
            plot_k: bool = False
                If set to `True`, the algorithm will plot the inertia trendline for every `k` that has been explored.
                
            show_examples: bool = False
                If set to `True`, the algorithm will display 3 examples of each output cluster once the elbow has been found.

            verbose: bool = False
                If set to `True`, prints a message on the terminal specifying the clustering termination condition.
        
        
        -- Dimensionality reduction (optional) --
        
            All these parameters are used with their standard meaning in scikit-learn. Refer to the package's documentation.
            ```
            reducer: Dict[str, Any]
                n_components: int = 200
                n_iter: int = 20
                random_state: int = None
                ...
            ```
        
        """
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
        if self.reducer:
            X = self._reducer.fit_transform(X)
        else:
            X = np.asarray(X.todense())
        return X
    
    def __getattr__(self, key: str) -> Any:
        return self.__dict__[key]

    
    def __find_best_k(self, X: np.ndarray) -> Tuple[int, KMeans]:
        ks, inertias = [], []
        prev_inertia = None
    
        min_k, max_k = self.range
        for k in range(min_k, max_k + 1):
            if k >= X.shape[0]:
                if self.verbose:
                    print(f"Stopping early at k={k} since it's "
                          "equal to the number of input documents.")
                break
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
                    if self.verbose:
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
        """
        Transforms input text X to a numerical vector using TF-IDF Vectorizer,
        and possibly applying SVD dimensionality reduction.
        
        Args:
            X: Iterable of strings, representing text data.

        Returns:
            NumPy array of vectorized text data.
        """
        _X = self.__encode(X)
        return _X


    def __call__(self, X: Iterable[str]) -> Iterable[int]:
        """
        Function that fits model on input data X.
        
        Parameters
        ----------
        X: Iterable[str]
            Input text data that needs to be clustered.

        Returns
        -------
        Iterable[int]
            Cluster labels for each data in X.
        """
        _X = self.__encode(X)
        best_k, best_clustering = self.__find_best_k(_X)
        if self.plot_density:
            self.__plot_density(best_k, best_clustering)
        if self.show_examples:
            self.__show_examples(X, best_k, best_clustering)
        return best_clustering.labels_
    

    def __plot_density(self, best_k: int, best_clustering: KMeans) -> None:
        """
        Plots a density graph representing the distribution of data among clusters.
        
        Parameters
        ----------
        best_k: int
            Optimal number of clusters.

        best_clustering: KMeans
            KMeans model with optimal number of clusters.
        """
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
        """
        Prints representatives from each clusters.
        
        Parameters
        ----------
        X: Iterable[str]
            List of texts.

        best_k: int
            Optimal number of clusters.

        best_clustering: KMeans
            KMeans model with optimal number of clusters.
        """
        for cluster_num in range(best_k):
            samples = np.where(best_clustering.labels_ == cluster_num)[0]
            if len(samples) > 3:
                samples = np.random.choice(samples, 3)
            for sample in samples:
                print(f"{cluster_num + 1}: {X.iloc[sample]}")


testmod()

if __name__ == "__main__":

    rows = [
      'one text',
      'another text',
      'a similar text',
      'this sentence',
      'fourth sentence',
      'fifth sentence',
    ]
    df = pd.DataFrame(rows, columns=['text'])
    
    cls = Clustexts(
        reducer={},
        show_examples=True,
        min_gain=0.001,
        vectorizer={'min_df': 0.0}
    )
    df['cluster'] = cls(df['text'])


    params = PARAMS.copy()
    params['reducer'] = dict([])
    eklus = Clustexts(**params)
    print(eklus)