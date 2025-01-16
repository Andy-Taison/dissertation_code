import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Suppresses pycharm warning for umap not being in requirements, umap is part of umap-learn package
# noinspection PyPackageRequirements
import umap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..model.model import VAE
from ..config import DEVICE, RANDOM_STATE, PLOT_DIR
import warnings
# Suppress warnings from sklearn and umap regarding future deprecations and expected behavior:
# - 'force_all_finite' deprecation (scikit-learn): 'check_array' function in scikit-learn uses it internally, 'force_all_finite' is scheduled to be replaced with 'ensure_all_finite'
# - 'n_jobs' override (umap-learn): UMAP overrides 'n_jobs' to 1 when a random_state is provided to ensure reproducibility
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

plt.style.use('dark_background')

def sample_latent_space(model: VAE, dataloader: DataLoader) -> torch.Tensor:
    """
    Extract latent space representations.

    :param model: Trained VAE model
    :param dataloader: DataLoader for forward pass
    :return: Latent space representations
    """
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for _, grid_data in dataloader:
            x = grid_data.to(DEVICE)
            z_mean, z_log_var = model.encoder(x)
            z = model.sampling(z_mean, z_log_var)
            latent_vectors.append(z.cpu())

    latent_vectors = torch.cat(latent_vectors)

    return latent_vectors


def train_pca(latent_vectors: torch.Tensor, n_components: int = 2) -> PCA:
    """
    Train PCA on latent vectors.

    :param latent_vectors: Latent vectors from training set
    :param n_components: Number of PCA components
    :return: Trained PCA model
    """
    pca_reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_reducer.fit(latent_vectors.numpy())

    return pca_reducer


def train_umap(latent_vectors: torch.Tensor, n_components: int = 2, n_neighbors: int = 15) -> umap.UMAP:
    """
    Train UMAP on latent vectors.

    :param latent_vectors: Latent vectors from training set
    :param n_components: Number of UMAP components
    :param n_neighbors: Number of neighbors for UMAP
    :return: Trained UMAP model
    """
    umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=RANDOM_STATE)
    umap_reducer.fit(latent_vectors.numpy())

    return umap_reducer


def transform_latent_space(reducer, latent_vectors: torch.Tensor) -> np.ndarray:
    """
    Transform latent vectors using a trained PCA or UMAP model.

    :param reducer: Trained PCA or UMAP model
    :param latent_vectors: Latent vectors to transform
    :return: Transformed latent vectors
    """
    return reducer.transform(latent_vectors.numpy())


def kmeans_cluster(latent_vectors: np.ndarray, n_clusters: int = 3) -> tuple[KMeans, np.ndarray]:
    """
    Perform KMeans clustering on latent vectors.

    :param latent_vectors: Latent space representations
    :param n_clusters: Number of clusters
    :return: trained KMeans model, Cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    cluster_labels = kmeans.fit_predict(latent_vectors)

    return kmeans, cluster_labels


def find_optimal_k(latent_vectors: torch.Tensor, max_k: int = 10, title: str = "Elbow Method for Optimal k", filename: str | Path = "optimal_k"):
    """
    Plots a line plot.
    Use the elbow method to find the optimal number of clusters for KMeans.

    :param latent_vectors: Latent space representations
    :param max_k: Maximum number of clusters to evaluate
    :param title: Title of plot
    :param filename: Filename to save plot (without extension)
    """
    distortions = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans, _ = kmeans_cluster(latent_vectors.numpy(), n_clusters=k)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), distortions, marker='.')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.grid(True, alpha=0.3)

    # Save plot
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    plt.show()
    plt.close(fig)


def plot_latent_space(data: np.ndarray, cluster_labels: np.ndarray, title: str, filename: str | Path):
    """
    Plot the latent space after dimensionality reduction.

    :param data: Transformed latent space data
    :param cluster_labels: Cluster labels for the data points
    :param title: Title of plot
    :param filename: Filename to save plot without extension (save in 'PLOT_DIR' as specified in config)
    """
    fig = plt.figure(figsize=(8, 6))
    num_clusters = len(set(cluster_labels))
    cmap = plt.get_cmap("tab10", num_clusters)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap=cmap, alpha=0.8)
    plt.colorbar(scatter, label="Cluster", ticks=range(num_clusters))
    plt.clim(-0.5, num_clusters - 0.5)  # Center ticks to colourbar cells
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)

    # Save plot
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    plt.show()
    plt.close(fig)


def calculate_silhouette_score(latent_vectors: np.ndarray, cluster_labels: np.ndarray) -> float | None:
    """
    Calculate the silhouette score for clusterings.
    Use for evaluating cluster quality.
    A high silhouette score (close to 1) indicates well separated clusters.
    A low score (close to -1) suggests overlapping or poorly defined clusters.
    A score near 0 indicates overlapping clusters or points close to cluster boundaries.

    :param latent_vectors: Latent space representations
    :param cluster_labels: Cluster labels for each data point
    :return: Silhouette score
    """
    # Requires at least two clusters
    if len(set(cluster_labels)) < 2:
        print("Silhouette score cannot be calculated with fewer than 2 clusters.")
        return None

    score = silhouette_score(latent_vectors, cluster_labels)

    return score


def calculate_pairwise_distances(latent_vectors: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Compute pairwise distances (Euclidean) between all latent vectors.
    Can be used to measure diversity.
    High mean = greater diversity, and points are more spread out.
    Low mean = points are clustered closely together.
    Low std = points are relatively evenly distributed.
    High std = some regions are dense, others are sparse.


    :param latent_vectors: Latent space tensor of shape (N, D), where N is the number of latent vectors and D is the latent dimension
    :return: Pairwise distance matrix (symmetric with 0's on diagonal) of shape (N, N), where the element at [i, j] represents the distance between the i-th and j-th latent vectors, mean distance, std distance
    """
    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(latent_vectors, latent_vectors)

    return dist_matrix, dist_matrix.mean().item(), dist_matrix.std().item()


def analyse_latent_space(model, train_dataloader: DataLoader, val_dataloader: DataLoader, k: int, filename: str | Path = None, find_k: bool = False, title: str = None) -> dict | None:
    """
    Extract, train, reduce, and visualise latent space with PCA and UMAP.
    When 'find_k' is True, plots to find optimal K (using elbow method) instead of plotting PCA and UMAP visualisations.

    :param model: Trained VAE model
    :param train_dataloader: Training dataset dataLoader
    :param val_dataloader: Validation dataset dataLoader
    :param k: K value used for KMeans clustering
    :param filename: Filename to save plots without extension (saved in 'PLOT_DIR' as specified in config), when None, does not generate PCA or UMAP plots, appends PCA or UMAP plot type to filename
    :param find_k: When True, plots to find optimal K using elbow method, when False, plots latent space
    :param title: Title for plots (appended with plot specific information)
    :return: When 'find_k' is False, returns a dictionary of silhouette and pairwise metrics along with k value used
    """
    # Extract latent vectors
    train_latent = sample_latent_space(model, train_dataloader)
    val_latent = sample_latent_space(model, val_dataloader)

    # Train PCA and UMAP
    pca_reducer = train_pca(train_latent)
    umap_reducer = train_umap(train_latent)

    # Reduce latent space using PCA and UMAP
    val_pca_data = transform_latent_space(pca_reducer, val_latent)
    val_umap_data = transform_latent_space(umap_reducer, val_latent)

    if find_k:
        # Elbow method to find k
        print("Performing elbow method to find optimal k...")
        if title:
            find_optimal_k(val_latent, title=title, filename=filename)
        else:
            find_optimal_k(val_latent, filename=filename)

        return None
    else:
        # KMeans clustering
        pca_kmeans_model, pca_cluster_labels = kmeans_cluster(val_pca_data, n_clusters=k)
        umap_kmeans_model, umap_cluster_labels = kmeans_cluster(val_umap_data, n_clusters=k)

        if filename:
            pca_filename = f"{filename}_pca"
            umap_filename = f"{filename}_umap"
            title = title if title else "Latent Space"
            # Plot
            print(f"Plotting latent space using PCA and UMAP with K={k}...")
            plot_latent_space(val_pca_data, pca_cluster_labels, f"{title} - Validation Set (PCA)", pca_filename)
            plot_latent_space(val_umap_data, umap_cluster_labels, f"{title} - Validation Set (UMAP)", umap_filename)

        # Calculate Silhouette Score
        pca_silhouette = calculate_silhouette_score(val_pca_data, pca_cluster_labels)
        umap_silhouette = calculate_silhouette_score(val_umap_data, umap_cluster_labels)

        print("\nLatent analysis metrics:")
        print(f"Silhouette score (PCA): {pca_silhouette:.4f}")
        print(f"Silhouette score (UMAP): {umap_silhouette:.4f}\n")

        # Calculate Euclidean pairwise distance
        distances, mean_dist, std_dist = calculate_pairwise_distances(val_latent)
        print(f"Unique distances:\n{distances.unique()}\n")
        print(f"Max distance: {distances.max()}")
        print(f"Mean pairwise distance: {mean_dist:.4f}, Standard deviation: {std_dist:.4f}")

        return {
            "pca_sil": pca_silhouette,
            "umap_sil": umap_silhouette,
            "mean_pd": mean_dist,
            "std_pd": std_dist,
            "k_used": k
        }
