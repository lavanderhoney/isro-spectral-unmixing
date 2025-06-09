#%%
"""
Implements the clustering algorithms:
- KMeans (using faiss-cpu)
- GMM (Gaussian Mixture Model)
"""
import faiss
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Literal, Tuple, Optional
from sklearn.mixture import GaussianMixture
from matplotlib import colors as mcolors
from sklearn.preprocessing import StandardScaler
from dimension_reduction.latent_vectors import extract_latent_vectors
import argparse
# import matplotlib
# matplotlib.use('TkAgg')
#%%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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

def plot_and_eval(H, k, labels_image, algorithm_name, img_type: Literal['original', 'latent']='original', model_name: Optional[Literal['vae', 'ss-vae']]=None ):
    """
    Plot the clustering results overlay and evaluate using silhouette score.
    """
    colors = [
        '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00',
        '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB'
    ][:k]
    cmap = mcolors.ListedColormap(colors)

    # Compute effective dimensions from label length
    effective_rows, effective_cols = labels_image.shape[0], labels_image.shape[1] #change to make it dynamic with patch size

    # Create an RGB image from cluster labels
    cluster_image = labels_image.reshape(effective_rows, effective_cols)
    print("Cluster image shape:", cluster_image.shape)
    print("Labels shape: ", labels_image.shape)
    rgb_image = cmap(cluster_image / (k - 1))[:, :, :3]  # Drop alpha channel

    # Normalize a grayscale band (e.g., band 30) — original resolution
    gray_band = H[30, :, :]
    gray_norm = (gray_band - gray_band.min()) / (gray_band.max() - gray_band.min())

    rows, cols = H.shape[1], H.shape[2]
    # Center crop grayscale to match effective size
    start_r = (rows - effective_rows) // 2
    start_c = (cols - effective_cols) // 2
    gray_cropped = gray_norm[start_r:start_r + effective_rows, start_c:start_c + effective_cols]
    print("Gray cropped shape:", gray_cropped.shape)
    # Overlay grayscale and cluster colors
    alpha = 0.5
    overlay = (1 - alpha) * np.stack([gray_cropped] * 3, axis=2) + alpha * rgb_image

    # Plot
    plt.figure(figsize=(5, 10))
    plt.imshow(overlay)
    plt.title(f"{img_type} {algorithm_name} Clustering Results (k={k})")
    plt.axis('off')
    plt.show()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if model_name:
        plt.savefig(f"{img_type}_{model_name}_{algorithm_name}_clustering_k{k}_{timestamp}.png", bbox_inches='tight')
    else:
        plt.savefig(f"{img_type}_{algorithm_name}_clustering_k{k}_{timestamp}.png", bbox_inches='tight')

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
    print("Data loaded and reshaped:", X_flat.shape)
    kmeans_labels, score = kmeans_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    print("KMeans clustering completed.", kmeans_labels.shape)
    print("Silhouette Score:", score)
    _ = plot_and_eval(H, 4, kmeans_labels, "KMeans")
    
    
#%%
    gmm_labels = gmm_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
    print("GMM clustering completed.", gmm_labels.shape)
    plot_and_eval(H, 4, gmm_labels, "GMM")

#%%
    scaler = StandardScaler()
    X_flat_norm = scaler.fit_transform(X_flat)
    # latent_vectors = extract_latent_vectors('vae', 'src/models/model_state_vae_0609_034523.pth', X_flat_norm)
    # print("Latent vectors extracted.", latent_vectors.shape)
    
    # latent_kmeans_labels_vae, score = kmeans_clustering(latent_vectors, n_clusters=4, rows=rows, cols=cols)
    # print("KMeans clustering on latent vectors completed.", latent_kmeans_labels_vae.shape)
    # print("Silhouette Score for latent vectors:", score)
    # __ = plot_and_eval(H, 4, latent_kmeans_labels_vae, "KMeans Latent", img_type='latent', model_name='vae')
    
    # print("ARI Score:", adjusted_rand_score(kmeans_labels.flatten(), latent_kmeans_labels_vae.flatten()))
    # print("NMI Score:", normalized_mutual_info_score(kmeans_labels.flatten(), latent_kmeans_labels_vae.flatten()))
    
    #%%

    latent_vectors = extract_latent_vectors('ss-vae', 'src/models/model_state_ss_vae_0609_035249.pth', X_flat)
    print("Latent vectors extracted.", latent_vectors.shape)
    
    latent_kmeans_labels_ssvae, score = kmeans_clustering(latent_vectors, n_clusters=4, rows=997, cols=246)
    print("KMeans clustering on latent vectors completed.", latent_kmeans_labels_ssvae.shape)
    print("Silhouette Score for latent vectors:", score)
    __ = plot_and_eval(H, 4, latent_kmeans_labels_ssvae, "KMeans Latent", img_type='latent', model_name='ss-vae')
    orig_rows, orig_cols = H.shape[1], H.shape[2]
    latent_rows, latent_cols = latent_kmeans_labels_ssvae.shape
    row_margin = (orig_rows - latent_rows) // 2
    col_margin = (orig_cols - latent_cols) // 2

    cropped_kmeans_labels = kmeans_labels[row_margin:row_margin + latent_rows,
                          col_margin:col_margin + latent_cols]

    print("ARI Score:", adjusted_rand_score(cropped_kmeans_labels.flatten(), latent_kmeans_labels_ssvae.flatten()))
    print("NMI Score:", normalized_mutual_info_score(cropped_kmeans_labels.flatten(), latent_kmeans_labels_ssvae.flatten()))
# %%
