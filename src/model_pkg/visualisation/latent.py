import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Suppresses pycharm warning for umap not being in requirements, umap is part of umap-learn package
# noinspection PyPackageRequirements
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerTuple
import cv2
from pathlib import Path
from ..model.model import VAE
from ..config import DEVICE, RANDOM_STATE, PLOT_DIR, PLOT
import warnings
# Suppress warnings from sklearn and umap regarding future deprecations and expected behavior:
# - 'force_all_finite' deprecation (scikit-learn): 'check_array' function in scikit-learn uses it internally, 'force_all_finite' is scheduled to be replaced with 'ensure_all_finite'
# - 'n_jobs' override (umap-learn): UMAP overrides 'n_jobs' to 1 when a random_state is provided to ensure reproducibility
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


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
            z_mean, z_log_var, _ = model.encoder(x)
            z = model.sampling(z_mean, z_log_var)
            latent_vectors.append(z.cpu())

    latent_vectors = torch.cat(latent_vectors)

    return latent_vectors


def normalise_latent(latent_vector: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Fits a scaler to the passed latent vector.

    :param latent_vector: Numpy latent vector to fit scaler to and normalise
    :return: Normalised vector, Fitted scaler
    """
    # Normalise
    scaler = StandardScaler()
    normalised_latent = scaler.fit_transform(latent_vector)

    return normalised_latent, scaler


def train_pca(latent_vectors: np.ndarray, n_components: int = 2) -> tuple[PCA, np.ndarray]:
    """
    Train PCA on latent vectors and transform.

    :param latent_vectors: Latent vectors from training set
    :param n_components: Number of PCA components
    :return: Trained PCA model, Transformed_data
    """
    pca_reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    transformed_data = pca_reducer.fit_transform(latent_vectors)

    return pca_reducer, transformed_data


def train_umap(latent_vectors: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> tuple[umap.UMAP, np.ndarray]:
    """
    Train UMAP on latent vectors and transform.

    :param latent_vectors: Latent vectors from training set
    :param n_components: Number of UMAP components
    :param n_neighbors: Number of neighbors for UMAP
    :return: Trained UMAP model, Transformed_data
    """
    umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=RANDOM_STATE)
    transformed_data = umap_reducer.fit_transform(latent_vectors)

    return umap_reducer, transformed_data


def transform_latent_space(reducer, latent_vectors: np.ndarray) -> np.ndarray:
    """
    Transform latent vectors using a trained PCA or UMAP model.

    :param reducer: Trained PCA or UMAP model
    :param latent_vectors: Latent vectors to transform
    :return: Transformed latent vectors
    """
    return reducer.transform(latent_vectors)


def kmeans_cluster(latent_vectors: np.ndarray, n_clusters: int = 5) -> tuple[KMeans, np.ndarray]:
    """
    Perform KMeans clustering on latent vectors.

    :param latent_vectors: Latent space representations
    :param n_clusters: Number of clusters
    :return: trained KMeans model, Cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    cluster_labels = kmeans.fit_predict(latent_vectors)

    return kmeans, cluster_labels


def find_optimal_k(latent_vectors: np.ndarray, max_k: int = 10, title: str = "Elbow Method for Optimal k", filename: str | Path = "optimal_k"):
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
        kmeans, _ = kmeans_cluster(latent_vectors, n_clusters=k)
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

    if PLOT:
        plt.show()
    plt.close(fig)


def plot_clustered_latent_space(data: np.ndarray, cluster_labels: np.ndarray, title: str, filename: str | Path):
    """
    Plot the latent space after dimensionality reduction with clustering labels.

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
    plt.clim(-0.5, num_clusters - 0.5)  # Centre ticks to colourbar cells
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

    if PLOT:
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

    # Normalise latent vectors
    train_latent_norm, scaler = normalise_latent(train_latent.numpy())  # Fit and transform
    val_latent_norm = scaler.transform(val_latent.numpy())  # Transform

    # Train PCA and UMAP
    pca_reducer, _ = train_pca(train_latent_norm)
    umap_reducer, _ = train_umap(train_latent_norm)

    # Reduce latent space using PCA and UMAP
    val_pca_data = transform_latent_space(pca_reducer, val_latent_norm)
    val_umap_data = transform_latent_space(umap_reducer, val_latent_norm)

    if find_k:
        # Elbow method to find k
        print("Performing elbow method to find optimal k...")
        if title:
            find_optimal_k(val_latent_norm, title=title, filename=filename)
        else:
            find_optimal_k(val_latent_norm, filename=filename)

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
            plot_clustered_latent_space(val_pca_data, pca_cluster_labels, f"{title} - Validation Set (PCA)", pca_filename)
            plot_clustered_latent_space(val_umap_data, umap_cluster_labels, f"{title} - Validation Set (UMAP)", umap_filename)

        # Calculate Silhouette Score
        pca_silhouette = calculate_silhouette_score(val_pca_data, pca_cluster_labels)
        umap_silhouette = calculate_silhouette_score(val_umap_data, umap_cluster_labels)

        print("Latent analysis metrics:")
        print(f"Silhouette score (PCA): {pca_silhouette:.4f}")
        print(f"Silhouette score (UMAP): {umap_silhouette:.4f}\n")

        # Calculate Euclidean pairwise distance
        distances, mean_dist, std_dist = calculate_pairwise_distances(torch.from_numpy(val_latent_norm))
        print(f"Unique distances:\n{distances.unique()}\n")
        print(f"Max distance: {distances.max():.4f}")
        print(f"Mean pairwise distance: {mean_dist:.4f}, Standard deviation: {std_dist:.4f}")

        return {
            "pca_sil": pca_silhouette,
            "umap_sil": umap_silhouette,
            "pairwise_mean": mean_dist,
            "pairwise_std": std_dist,
            "k_used": k
        }


def plot_pca_eigenvectors(ax: plt.axes, pca, latent_2d_vectors: np.ndarray, scale_factor: float = 3.0) -> tuple[list[patches.FancyArrow], np.ndarray]:
    """
    Plots PCA eigenvectors.

    :param ax: Axis to plot on
    :param pca: Trained PCA model
    :param latent_2d_vectors: Latent 2D vectors used in PCA
    :param scale_factor: Scaling for arrow length
    :return: FancyArrow object to be used in legend
    """
    mean_vec = latent_2d_vectors.mean(axis=0)
    eigenvectors = pca.components_  # Each row = eigenvector (principal component direction), each column corresponds to an original feature
    eigenvalues = pca.explained_variance_  # Variance explained by each of the principal components
    eigenvalues_std = np.sqrt(eigenvalues)

    legend_entry = []
    colours = ["red", "yellow"]

    for i in range(2):
        vec = eigenvectors[i] * eigenvalues_std[i] * scale_factor  # Eigenvectors * standard deviation (to represent real data spread) * scaling factor for visibility in the plot

        legend_entry.append(ax.arrow(mean_vec[0], mean_vec[1], vec[0], vec[1], color=colours[i], width=0.06, head_width=0.3, alpha=0.8))  #, label="Eigenvector PC1") if i == 0 else ax.arrow(mean_vec[0], mean_vec[1], vec[0], vec[1], color='y', width=0.06, head_width=0.25, alpha=0.8, label="Eigenvector PC2")

    return legend_entry, eigenvalues_std


def fit_ellipse(ax: plt.Axes, ds: np.ndarray, colour):
    # Fit ellipse using opencv
    ellipse = cv2.fitEllipse(ds)

    # Obtain data for calculating area
    (x, y), (major, minor), angle = ellipse
    print(f"major: {major}")
    print(f"minor: {minor}")
    print(f"angle: {angle}")

    # OpenCV returns width and height as minor/major axes, but Matplotlib expects (width, height)
    # Though it doesn't look like it would fit in any orientation

    mat_ellipse = patches.Ellipse((x, y), major, minor, angle=angle, color=colour, fill=None)
    ax.add_patch(mat_ellipse)


def plot_latent_space_evaluation(latent_2d_vectors: np.ndarray, dataset_labels: list[str], centres: dict[str, np.ndarray], title: str, pca_model=None, filename: str = None, x_ax_min: int = 0, y_ax_min: int = 0, plot_set_colour: str = "all"):
    unique_labels = sorted(set(dataset_labels))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    colour_idx = {}

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(6, 6)
    ax = fig.add_subplot(gs[:, :-1])  # Main plot
    legend_ax = fig.add_subplot(gs[:, -1])  # Legend subplot

    legend_handles = []  # Symbols
    legend_labels = []  # Text labels

    for i, label in enumerate(unique_labels):
        if plot_set_colour.lower() == label.lower():
            alpha = 1.0
            colour_idx[label] = cmap(i)
        elif plot_set_colour.lower() == "all":
            alpha = 0.7
            colour_idx[label] = cmap(i)
        else:
            alpha = 0.5
            colour_idx[label] = "gray"

        display_label = ' '.join(word.capitalize() for word in label.split('_'))
        idxs = np.array(dataset_labels) == label
        ds = latent_2d_vectors[idxs]

        # Scatter plot for points
        dots = ax.scatter(ds[:, 0], ds[:, 1], label=display_label, alpha=alpha, color=colour_idx[label], linewidth=0.5)
        legend_handles.append(dots)
        legend_labels.append(display_label)

        # fit_ellipse(ax, ds, colour_idx[label])  # Does not function correctly

    # Centre of masses - separate for loop for tidy legend
    for label, centre in centres.items():
        display_label = ' '.join(word.capitalize() for word in label.split('_'))
        crosses = ax.scatter(centres[label][0], centres[label][1], marker='X', s=150, edgecolors='white', color=colour_idx[label], label=f"{display_label} Centre")
        legend_handles.append(crosses)
        legend_labels.append(display_label)

    # Plot Eigenvectors
    if pca_model is not None:
        handles, eigenvals = plot_pca_eigenvectors(ax, pca_model, latent_2d_vectors)
        legend_handles.extend(handles)
        legend_labels.extend([f"Eigenvector PC1, $\\sqrt{{\\lambda}} = {eigenvals[0]:.4f}$", f"Eigenvector PC2, $\\sqrt{{\\lambda}} = {eigenvals[1]:.4f}$"])

    # Add distances between centre of masses to legend
    dummy, = ax.plot([], [], ' ', label="\nCentre Euclidean Distances:")
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nCentre Euclidean Distances:")
    keys = list(centres.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):  # Only consider higher indices as don't need both directions
            label_a = keys[i]
            label_b = keys[j]
            centre_a = centres[label_a]
            centre_b = centres[label_b]
            distance = np.linalg.norm(centre_a - centre_b)

            # Create a dummy ax for symbols
            a_legend_key = ax.scatter([], [], marker='X', s=150, edgecolors='white', color=colour_idx[label_a])
            arrow_key = ax.scatter([], [], marker="$-$", color='white')
            b_legend_key = ax.scatter([], [], marker='X', s=150, edgecolors='white', color=colour_idx[label_b])

            legend_handles.append((a_legend_key, arrow_key, b_legend_key))  # Symbols

            # Add distance as label
            distance = f"{distance:.4f}"
            legend_labels.append(distance)  # Distance

    fig.suptitle(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Make axes symmetric around 0
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    new_x_limit = max(abs(x_min), abs(x_max), x_ax_min)
    new_y_limit = max(abs(y_min), abs(y_max), y_ax_min)
    ax.set_xlim(-new_x_limit, new_x_limit)
    ax.set_ylim(-new_y_limit, new_y_limit)

    ax.axhline(0, alpha=0.3)  # Add grid line at y=0
    ax.axvline(0, alpha=0.3)  # Add grid line at x=0

    # Add legend to subplot
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_handles, labels=legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)})

    # Save plot
    if filename is None:
        filename = title.lower().replace(" ", "_").replace("_-_", "-").replace(":", "")
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    if PLOT:
        plt.show()
    plt.close(fig)


def compute_centres(latent_2d_vectors: np.ndarray, dataset_labels: list[str]) -> dict[str, np.ndarray]:
    """
    Computes mean centre of mass per dataset.

    :param latent_2d_vectors: 2D latent vectors
    :param dataset_labels: Labels list corresponding to latent_2d_vectors
    :return: Centre of mass dictionary with dataset label as key
    """
    unique_labels = sorted(set(dataset_labels))  # Alphabetical and consistent mapping
    centres = {}

    for label in unique_labels:
        idxs = np.array(dataset_labels) == label  # Bool array
        centres[label] = latent_2d_vectors[idxs].mean(axis=0)

    return centres


def evaluate_latent_vectors(latent_vectors: np.ndarray, dataset_labels: list[str], title: str = None, x_ax_min: int = 0, y_ax_min: int = 0, plot_set_colour="all"):
    # Normalise
    norm_latents, _ = normalise_latent(latent_vectors)
    print("#"*100)

    # Apply PCA
    # Global
    pca, pca_data = train_pca(norm_latents)
    # Per dataset
    unique_sets = set(dataset_labels)
    local_pca_data = {}
    for ds in unique_sets:
        mask = [True if label == ds else False for label in dataset_labels]
        ds_norm_latent = norm_latents[mask]
        temp_dict = {"pca_model": (train_pca(ds_norm_latent))[0], "data": (train_pca(ds_norm_latent))[1]}
        local_pca_data[ds] = temp_dict

    # Apply UMAP
    # Global
    _, umap_data = train_umap(norm_latents)
    # Per dataset
    local_umap_data = {}
    for ds in unique_sets:
        mask = [True if label == ds else False for label in dataset_labels]
        ds_norm_latent = norm_latents[mask]
        _, local_umap_data[ds] = train_pca(ds_norm_latent)

    # Compute centres of mass
    pca_centres = compute_centres(pca_data, dataset_labels)
    umap_centres = compute_centres(umap_data, dataset_labels)

    if title is None:
        title_suffix = ""
    else:
        title_suffix = f" - {title}"

    # Plot
    plot_latent_space_evaluation(pca_data, dataset_labels, pca_centres, f"PCA Projection{title_suffix}", pca_model=pca, x_ax_min=x_ax_min, y_ax_min=y_ax_min, plot_set_colour=plot_set_colour)
    plot_latent_space_evaluation(umap_data, dataset_labels, umap_centres, f"UMAP Projection{title_suffix}", x_ax_min=x_ax_min, y_ax_min=y_ax_min, plot_set_colour=plot_set_colour)
