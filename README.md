# Clustexts

**Last updated: 13th March, 2025**Performs k-means clustering on a collection of texts. It automates the selection of `k` by running the elbow method implicitly. The algorithm only expects the range of minimum and maximum values for `k` (default to 2 and 20, respectively).
    
Texts are encoded using a TFIDF Bag-of-Words representation. Optionally, Truncated Singular Value Decomposition can be used to reduce the dimensionality of the resulting matrix and project the topology onto an embedded space, for improved data compression and schema generalization.
    
The call to an initialized and fitted instance of this object returns an iterator containing the cluster identifiers associated with each input document.## DependenciesEnsure you have the following packages installed:
```
matplotlib==3.10.1
numpy==2.2.3
pandas==2.2.3
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
tqdm==4.67.1
```
## Usage

Given the following sample toy **dataset**:```rows = [  'one text',  'another text',  'this sentence',  'fourth sentence',  'fifth sentence',]df = pd.DataFrame(rows, columns=['text'])```

an instance of the `Clustext` **class** can be **instantiated** as shown below:

```cls = Clustexts(  reducer={},  range = (2, 10),  min_gain=0.001,  vectorizer={'min_df': 0.0})
```

and **clustering** can then be performed as shown in the line below:

```df['cluster'] = cls(df['text'])```

## Parameters

### Clustering
- `range: Tuple[int, int] = (2, 20)`: Specifies the minimum and maximum values of `k` to explore when applying the elbow method.
- `min_size: int = 0`: The minimum cluster size to be accepted. If reached, the clustering stops.
- `min_gain: float = 0.03`: The minimum relative improvement for the clustering to continue running (as a percentage of the inertia).

### Vectorization (required) & Dimensionality reduction (optional)

Refer to the scikit-learn's documentation for the `TfidfVectorizer` and the `TruncatedSVD` classes.

### Reporting (optional)

- `plot_density: bool = False`: If set to `True`, the system will plot cluster densities (number of documents in each cluster).
- `plot_k: bool = False`: If set to `True`, the algorithm will plot the inertia trendline for every `k` that has been explored.
- `show_examples: bool = False`: If set to `True`, the algorithm will display 3 examples of each output cluster after the elbow has been found.
- `verbose: bool = False`: If set to `True`, prints a message on the terminal specifying the clustering termination condition.

## Methods

- `encode(X: Iterable[str]) -> np.ndarray`: Transforms input text X to a numerical vector using TF-IDF Vectorizer, and optionally applies SVD dimensionality reduction.
- `__call__(self, X: Iterable[str]) -> Iterable[int]`: fits model on input data X.