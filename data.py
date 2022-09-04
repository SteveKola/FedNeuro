# the self supervised embeddings should be computed once because it takes a long time!
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
from time import time
import numpy as np

digits = load_digits(n_class=10)
X, y = digits.data, digits.target
# data{ndarray, dataframe} of shape (1797, 64)
# target: {ndarray, Series} of shape (1797,)
n_samples, n_features = X.shape
n_neighbors = 30


# n_components - Preferred dimensionality of the projected space

embeddings = {
    "Truncated SVD embedding": TruncatedSVD(n_components=1),
    "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
        n_components=1
    ),
    "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=1),
    "Standard LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="standard"
    ),
    "Modified LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="modified"
    ),
    "Hessian LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="hessian"
    ),
    "LTSA LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="ltsa"
    ),
    "MDS embedding": MDS(n_components=1, n_init=1, max_iter=120, n_jobs=2),
    "Random Trees embedding": make_pipeline(
        RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
        TruncatedSVD(n_components=1),
    ),
    "Spectral embedding": SpectralEmbedding(
        n_components=1, random_state=0, eigen_solver="arpack"
    ),
    "t-SNE embedding": TSNE(
        n_components=1,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    ),
    "NCA embedding": NeighborhoodComponentsAnalysis(
        n_components=1, init="pca", random_state=0
    ),
}


projections, timing = {}, {}
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    # print(f"Computing {name}...")
    start_time = time()
    projections[name] = transformer.fit_transform(data, y)
    # print(projections[name]) # 1 row with 1797 items - each item in the array is an array
    timing[name] = time() - start_time


all_items = []
for name in timing:
    all_items.append(projections[name])
    
np.save("data.npy", all_items)