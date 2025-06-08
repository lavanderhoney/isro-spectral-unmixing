#%%
"""
Implements the clustering algorithms:
- KMeans (using faiss-cpu)
- GMM (Gaussian Mixture Model)
"""
import faiss
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple
from sklearn.mixture import GaussianMixture
from matplotlib import colors as mcolors
from sklearn.preprocessing import StandardScaler
from dimension_reduction.latent_vectors import extract_latent_vectors
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
    score = silhouette_coefficient(fkmeans.index, X)
    return labels_image, score

def gmm_clustering(X, n_clusters, rows, cols):
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(X)
    # Reshape labels to original spatial dimensions if needed
    labels_image = labels.reshape(rows, cols)
    return labels_image

def silhouette_coefficient(cluster_index, samples) -> float:
    # At each data point, we calculate the distance to the cluster center in which the data point belongs (referred to as a), as well as the distance to the second best cluster center (referred to as b). Here, the second best cluster refers
    # to the closest cluster that is not the current data point’s cluster. Then based, on these two distances a and b, the
    # silhouette s of that data point is calculated as s=(b-a)/max(a,b). The mean of all s is the s coefficient and measures
    # quality of the clustering for this nr of k
    #
    # This is done by simply calculating the silhouette coefficient over a range of k, and identifying the peak as the optimum K.
    distance, _ = cluster_index.search(samples, 2)

    # a = distance[:, 0], b = distance[:, 1]
    # s = (b - a) / np.max(a, b)
    s = (distance[:, 1] - distance[:, 0]) / np.max(distance, 1)

    # A score of 1 denotes the best, meaning that the data point i is very compact within the cluster to which it belongs
    # and far away from the other clusters. The worst value is -1. Values near 0 denote overlapping clusters.
    return s.mean()

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
      
    return rgb_image

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Clustering algorithms for spectral unmixing.")
    # parser.add_argument("--data_path", type=str, default="/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz",
    #                     help="Path to the input data file.")
    # parser.add_argument("--radius", type=float, default=1.0,
    #                     help="Radius for DBSCAN clustering.")
    # args = parser.parse_args()
    # data_path = args.data_path
    data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz"
    try:
        data = np.load(data_path)
        H = data["den_refl_data"]  # Expected shape: (bands, rows, cols)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        exit(1)
    except KeyError:
        print(f"Error: 'refl_data' key not found in {data_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    H_t = np.moveaxis(H, 0, 2)  # Shape: (rows, cols, bands)
    H_t = H_t.astype('float32')
    rows, cols, bands = H_t.shape
    X_flat = H_t.reshape(rows*cols, bands)
    
    kmeans_labels, score = kmeans_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    print("KMeans clustering completed.", kmeans_labels.shape)
    print("Silhouette Score:", score)
    plot_and_eval(H, 4, kmeans_labels, "KMeans")
    
    
#%%
    # gmm_labels = gmm_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    # print("GMM clustering completed.", gmm_labels.shape)
    # plot_and_eval(H, 4, gmm_labels, "GMM")

#%%
    scaler = StandardScaler()
    X_flat_norm = scaler.fit_transform(X_flat)
    latent_vectors = extract_latent_vectors('vae', '/teamspace/studios/this_studio/isro-spectral-unmixing/models/vae_model_0608_102826.pth', X_flat_norm)
    print("Latent vectors extracted.", latent_vectors.shape)
    
    latent_kmeans_labels, score = kmeans_clustering(latent_vectors, n_clusters=4, rows=rows, cols=cols)
    print("KMeans clustering on latent vectors completed.", latent_kmeans_labels.shape)
    print("Silhouette Score for latent vectors:", score)
    __ = plot_and_eval(H, 4, latent_kmeans_labels, "KMeans Latent", img_type='latent')

#%%
    latent_kmeans_labels, score = kmeans_clustering(latent_vectors, n_clusters=2, rows=rows, cols=cols)
    print("KMeans clustering on latent vectors completed.", latent_kmeans_labels.shape)
    print("Silhouette Score for latent vectors (k=2):", score)
    __ = plot_and_eval(H, 2, latent_kmeans_labels, "KMeans Latent", img_type='latent')
    
    #%%
    latent_kmeans_labels, score = kmeans_clustering(latent_vectors, n_clusters=3, rows=rows, cols=cols)
    print("KMeans clustering on latent vectors completed.", latent_kmeans_labels.shape)
    print("Silhouette Score for latent vectors (k=3):", score)
    __ = plot_and_eval(H, 3, latent_kmeans_labels, "KMeans Latent", img_type='latent')
    
    #%%

    latent_vectors = extract_latent_vectors('ss-vae', '/teamspace/studios/this_studio/isro-spectral-unmixing/models/model_state_ss_vae_0608_183236.pth', X_flat_norm)
    print("Latent vectors extracted.", latent_vectors.shape)
    
    latent_kmeans_labels, score = kmeans_clustering(latent_vectors, n_clusters=4, rows=rows, cols=cols)
    print("KMeans clustering on latent vectors completed.", latent_kmeans_labels.shape)
    print("Silhouette Score for latent vectors:", score)
    __ = plot_and_eval(H, 4, latent_kmeans_labels, "KMeans Latent", img_type='latent')
# %%
