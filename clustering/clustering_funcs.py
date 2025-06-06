#%%
"""
Implements the clustering algorithms:
- KMeans (using faiss-cpu)
- DBSCAN
- Heirarchical Clustering
- GMM (Gaussian Mixture Model)
- Spectral Clustering
"""
import faiss
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from matplotlib import colors as mcolors
from sklearn.preprocessing import StandardScaler
import argparse

#%%
def kmeans_clustering(X: np.ndarray, n_clusters: int, rows: int, cols: int):
    fkmeans = faiss.Kmeans(d=X.shape[1],
                      k=n_clusters,
                      niter=20,
                      )
    fkmeans.train(X.astype(np.float32))

    distances, labels = fkmeans.index.search(X, 1) # type: ignore
    labels = labels.ravel()  # Flatten to 1D array of shape (rows * cols,)

    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    return labels_image

def dbscan_clustering(X: np.ndarray, rows:int, cols:int, radius: float =0.5)->Tuple[np.ndarray, int]:
    """
    X: np.ndarray
        The input data to cluster, shape (n_samples, n_features).
    radius: float (eps in DBSCAN)
    """
    db = DBSCAN(eps=radius, n_jobs=-1).fit(X)
    labels = db.labels_
    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #ignore noise points
    n_noise = list(labels).count(-1)
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    return labels_image, n_clusters

def hierarchical_clustering(X: np.ndarray, n_clusters: int,  rows: int, cols:int, linkage: Literal['ward','complete', 'average', 'single']='ward') -> np.ndarray:
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage) # type: ignore
    labels = hc.fit_predict(X)
    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    return labels_image

def gmm_clustering(X, n_clusters, rows, cols):
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(X)
    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    return labels_image

def spectral_clustering(X, n_clusters, rows, cols):
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_jobs=-1)
    labels = sc.fit_predict(X)
    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    return labels_image

def plot_and_eval(H, k, labels_image, algorithm_name, img_type: Literal['original', 'latent']='original'):
    """
    Plot the clustering results overlay and evaluate using silhouette score.
    """
    colors = [
        '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00',
        '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB'
    ][:k]
    cmap = mcolors.ListedColormap(colors)

    # Create an RGB image from cluster labels
    cluster_image = labels_image.reshape(H.shape[1], H.shape[2])
    rgb_image = cmap(cluster_image / (k - 1))[:, :, :3]  # Drop alpha channel

    print(rgb_image.shape)
    # Normalize a grayscale band (e.g., band 30)
    gray_band = H[30, :, :]
    gray_norm = (gray_band - gray_band.min()) / (gray_band.max() - gray_band.min())
    print(gray_norm.shape)
    alpha = 0.5  # 0 = original only, 1 = cluster overlay only
    overlay = (1 - alpha) * np.stack([gray_norm]*3, axis=2) + alpha * rgb_image
    
    plt.figure(figsize=(5, 10))
    plt.imshow(overlay)
    plt.title(f"{img_type} {algorithm_name} Clustering Results (k={k})")
    plt.axis('off')
    plt.show()
        # Flatten the labels for silhouette score calculation
    labels_flat = labels_image.flatten()
    if len(set(labels_flat)) > 1:  # Ensure at least 2 clusters for silhouette score
        score = silhouette_score(labels_flat.reshape(-1, 1), labels_flat)
        print(f"Silhouette Score for {algorithm_name}: {score:.4f}")
    else:
        print("Silhouette Score cannot be computed with less than 2 clusters.")
        
    return score, rgb_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering algorithms for spectral unmixing.")
    parser.add_argument("--data_path", type=str, default="/teamspace/studios/this_studio/isro-spectral-unmixing/data/reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz",
                        help="Path to the input data file.")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Radius for DBSCAN clustering.")
    args = parser.parse_args()
    data_path = args.data_path
    try:
        data = np.load(args.data_path)
        H = data["refl_data"]  # Expected shape: (bands, rows, cols)
    except FileNotFoundError:
        print(f"Error: File not found at {args.data_path}")
        exit(1)
    except KeyError:
        print(f"Error: 'refl_data' key not found in {args.data_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    H_t = np.moveaxis(H, 0, 2)  # Shape: (rows, cols, bands)
    H_t = H_t.astype('float32')
    rows, cols, bands = H_t.shape
    X_flat = H_t.reshape(rows*cols, bands)
    
    kmeans_labels = kmeans_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    print("KMeans clustering completed.", kmeans_labels.shape)
    dbscan_labels, n_clusters = dbscan_clustering(X_flat, rows=rows, cols=cols, radius=1.0)
    print("DBSCAN clustering completed.")
    #%%
    # hierarchical_labels = hierarchical_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    gmm_labels = gmm_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    spectral_labels = spectral_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    #%%
    # Plot and evaluate each clustering result
    plot_and_eval(H_t, 4, kmeans_labels, "KMeans")
    plot_and_eval(H_t, n_clusters, dbscan_labels, "DBSCAN")
    #%%
    # plot_and_eval(H_t, 4, hierarchical_labels, "Hierarchical Clustering")
    plot_and_eval(H_t, 4, gmm_labels, "GMM")
    plot_and_eval(H_t, 4, spectral_labels, "Spectral Clustering")