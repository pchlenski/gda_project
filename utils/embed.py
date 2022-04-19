import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.datasets import make_swiss_roll

def run_pca(data, n_dims, n_neighbors=None, **kwargs):
    """ Project down to n dimensions using PCA """
    pca = PCA(
        n_components=n_dims, 
        **kwargs
    )
    return pca.fit_transform(data)

def run_isomap(data, n_neighbors, n_dims, **kwargs):
    """ Project down to n dimensions using IsoMap """
    isomap = Isomap(
        n_neighbors=n_neighbors,
        n_components=n_dims,
        **kwargs
    )
    return isomap.fit_transform(data)

def run_lle(data, n_neighbors, n_dims, **kwargs):
    """ Project down to n dimensions using LLE """
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=n_dims,
        **kwargs
    )
    return lle.fit_transform(data)

def run_laplacian(data, n_neighbors, n_dims, **kwargs):
    """ Project to n dimensions using graph Laplacian """ 
    laplacian = SpectralEmbedding(
        n_neighbors=n_neighbors,
        n_components=n_dims,
        **kwargs
    )
    return laplacian.fit_transform(data)

def bakeoff(data, n_neighbors, n_dims):
    """ Uses PCA, IsoMap, LLE, and spectral embeddings """
    out = []
    for run_reduction in [run_pca, run_isomap, run_lle, run_laplacian]:
        out.append(run_reduction(data, n_neighbors=n_neighbors, n_dims=n_dims))
    return out

def swiss_roll_test(n_samples=100, n_neighbors=10):
    """ Generate a swiss roll and run the bakeoff """
    data_ambient, data_manifold = make_swiss_roll(n_samples=n_samples)
    embeddings = bakeoff(data_ambient, n_neighbors=n_neighbors, n_dims=2)

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    for k, (embedding, name) in enumerate(zip(embeddings, ["PCA", "IsoMap", "LLE", "Spectral"])):
        i,j = np.divmod(k,2)
        axs[i,j].scatter(embedding[:,0], embedding[:,1], c=data_manifold)
        axs[i,j].set_title(name)