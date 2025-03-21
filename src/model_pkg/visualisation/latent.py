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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from functools import partial
import shapely
from shapely.geometry import Point
from shapely.affinity import scale, rotate
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


def plot_pca_eigenvectors(ax: plt.axes, pca, scale_factor: float = 3.0) -> tuple[list[patches.FancyArrow], np.ndarray]:
    """
    Plots PCA eigenvectors.

    :param ax: Axis to plot on
    :param pca: Trained PCA model
    :param scale_factor: Scaling for arrow length
    :return: FancyArrow object to be used in legend
    """
    eigenvectors = pca.components_  # Each row = eigenvector (principal component direction), each column corresponds to an original feature
    eigenvalues = pca.explained_variance_  # Variance explained by each of the principal components
    eigenvalues_std = np.sqrt(eigenvalues)  # Standard deviation in same units as original data

    legend_entry = []
    colours = ["red", "orange"]

    for i in range(2):
        vec = eigenvectors[i] * eigenvalues_std[i] * scale_factor  # Eigenvectors * standard deviation (to represent real data spread) * scaling factor for visibility in the plot

        legend_entry.append(ax.arrow(0, 0, vec[0], vec[1], color=colours[i], width=0.06, head_width=0.3, alpha=0.8, zorder=4))  #, label="Eigenvector PC1") if i == 0 else ax.arrow(mean_vec[0], mean_vec[1], vec[0], vec[1], color='y', width=0.06, head_width=0.25, alpha=0.8, label="Eigenvector PC2")

    return legend_entry, eigenvalues_std


def getMinVolEllipse(points: np.ndarray, tolerance: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Minimum Volume Enclosing Ellipse (MVEE) using Khachiyan's Algorithm.

    Adapted from Minillinim. (n.d.). ellipsoid.py [Computer software]. GitHub. Retrieved March 15, 2025, from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py.
    Based on work by Moshtagh, N. (MATLAB Central File Exchange) and Scitbx mathematics library (CCTBX)

    :param points: (N, d) NumPy array of N points in d dimensions
    :param tolerance: Tolerance
    :return: Centre of ellipse, radii, rotation
    """
    (N, d) = np.shape(points)
    d = float(d)  # Features

    # Q is working array
    Q: np.ndarray = np.vstack([points.T, np.ones(N)])
    QT = Q.T

    # Initialise
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M: np.ndarray = np.diag(np.dot(QT, np.dot(np.linalg.inv(V), Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        max_M = M[j]
        step_size = (max_M - d - 1.0) / ((d + 1.0) * (max_M - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # Centre of ellipse
    centre: np.ndarray = np.dot(points.T, u)

    A = np.linalg.inv(
        np.dot(points.T, np.dot(np.diag(u), points)) - np.array([[a * b for b in centre] for a in centre])
    ) / d

    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    # # Experiment to check svd output format
    # # Create a known rotation matrix (45 degrees)
    # theta = np.radians(45)  # Convert 45 degrees to radians
    # # https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/#:~:text=R%3D(cos%CE%B8%E2%88%92sin,%CE%B8sin%CE%B8cos%CE%B8).&text=As%20of%20NumPy%20version%201.17,will%20be%20removed%20in%20future.
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
    #                           [np.sin(theta), np.cos(theta)]])
    # # Compute SVD
    # U, S, exp_rotate = np.linalg.svd(rotation_matrix)
    #
    # # Test point on the x-axis
    # point = np.array([1, 0])  # (1, 0)
    #
    # # Apply rotation matrix
    # rotated_point = np.dot(point, rotation_matrix)  # Matrix multiplication
    #
    # print(f"Original Point: {point}")
    # print(f"Rotated Point: {rotated_point}")  # (0.7, -0.7) clockwise rotation
    #
    # # Extract angle first in radians then converts to degrees
    # rotation_angle = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    # print(f"rotation_angle: {rotation_angle}")  # 45 degrees extracted ok

    return centre, radii, rotation


def fit_ellipse(ax: plt.Axes, data_points: np.ndarray, colour, alpha) -> tuple[np.ndarray, np.ndarray, np.float64]:
    """
    Fits and plots an ellipse to the dataset using the Minimum Volume Enclosing Ellipse (MVEE).

    :param ax: Matplotlib Axes object
    :param data_points: (N, 2) NumPy array of 2D data points
    :param colour: Colour of ellipse
    :param alpha: Opacity of ellipse
    :return: Centre, radii(major,minor), rotation angle (degrees)
    """
    # Fit ellipse
    centre, radii, rotation = getMinVolEllipse(data_points)

    # Angle parameter in radians for points
    u = np.linspace(0.0, 2.0 * np.pi, 100)

    # Ellipse points
    x = radii[0] * np.cos(u)
    y = radii[1] * np.sin(u)

    # Apply rotation to every point
    for i in range(len(x)):
        [x[i], y[i]] = np.dot([x[i], y[i]], rotation) + centre

    # Draw ellipse
    ax.plot(x, y, color=colour, linewidth=2, alpha=alpha, zorder=1)

    # Extract angle first in radians then converts to degrees
    rotation_angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))

    return centre, radii, rotation_angle


def compute_ellipse_overlap(ellipse1: shapely.geometry.polygon.Polygon, ellipse2: shapely.geometry.polygon.Polygon):
    """
    Compute the overlap area between two ellipses using Shapely.
    """
    # Compute overlapping region
    intersection = ellipse1.intersection(ellipse2)

    return intersection.area


def on_click(event, coordinates, labels):
    if event.inaxes:
        # Find nearest point
        distances = [(x - event.xdata) ** 2 + (y - event.ydata) ** 2 for x, y in coordinates]
        min_index = distances.index(min(distances))
        print(f"Robot ID: {labels[min_index]} at {coordinates[min_index]}")
        return


def plot_latent_space_evaluation(transformed_data: np.ndarray, dataset_labels: list[str], centres: dict[str, np.ndarray], title: str, pca_model=None, filename: str = None, x_ax_min: int = 0, y_ax_min: int = 0, plot_set_colour: str = "all", plot_idxs: list = None, all_robot_ids: list = None, annotate: bool = True):
    unique_labels = sorted(set(dataset_labels))  # Ensures consistent legend and colours

    # Custom colours
    custom_colours = ['#2ca02c', '#17becf', '#9467bd']  # Green, Blue, Purple
    # Create colourmap
    cmap = ListedColormap(custom_colours)
    colour_idx = {}

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(6, 6)
    ax = fig.add_subplot(gs[:, :-1])  # Main plot
    legend_ax = fig.add_subplot(gs[:, -1])  # Legend subplot

    legend_handles = []  # Symbols
    legend_labels = []  # Text labels

    # Plot points
    shapely_ellipses = {}
    for i, label in enumerate(unique_labels):
        # Set opacity and colour
        if plot_set_colour.lower() == label.lower():
            alpha = 1.0
            colour_idx[label] = cmap(i)
            zorder = 3
        elif plot_set_colour.lower() == "all":
            alpha = 0.7
            colour_idx[label] = cmap(i)
            zorder = 3
        else:
            alpha = 0.5
            colour_idx[label] = "gray"
            zorder = 1

        idxs = np.array(dataset_labels) == label  # Gets index based on original labels, order matches transformed_data
        ds = transformed_data[idxs]
        display_label = f"{' '.join(word.capitalize() for word in label.split('_'))}: {ds[:,0].size} Samples"

        # Scatter plot for points
        if plot_idxs is not None:
            idxs = [i for i in plot_idxs if idxs[i]]
            subset = transformed_data[idxs]
            x = subset[:, 0]
            y = subset[:, 1]
            dots = ax.scatter(x, y, label=display_label, alpha=alpha, color=colour_idx[label], linewidth=0.5, zorder=zorder)
        else:
            x = ds[:, 0]
            y = ds[:, 1]
            dots = ax.scatter(x, y, label=display_label, alpha=alpha, color=colour_idx[label], linewidth=0.5, zorder=zorder)
        legend_handles.append(dots)
        legend_labels.append(display_label)

        if annotate:
            dots.set_edgecolor("black")
            dots.set_linewidth(1)

        # Plot ellipse
        centre, radii, rotation = fit_ellipse(ax, ds, colour_idx[label], alpha)

        # Get shapely ellipse for overlaps and compute area, store for overlap calculation later
        ellipse, area = create_shapely_ellipse(centre, radii, rotation)
        shapely_ellipses[label] = {
            "colour": colour_idx[label],
            "area": area,
            "ellipse": ellipse
        }

    # Plot Eigenvectors
    if pca_model is not None:
        handles, eigenvals = plot_pca_eigenvectors(ax, pca_model)
        dummy, = ax.plot([], [], ' ')
        legend_handles.append(dummy)  # For correct spacing
        legend_handles.extend(handles)
        legend_labels.extend(["", f"Eigenvector PC1: $\\sqrt{{\\lambda}} = {eigenvals[0]:.4f}$", f"Eigenvector PC2: $\\sqrt{{\\lambda}} = {eigenvals[1]:.4f}$"])

    # Add distances between centre of masses to legend
    dummy, = ax.plot([], [], ' ')
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nCentre of Mass Euclidean Distances:")
    keys = list(centres.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):  # Only consider higher indices as don't need both directions
            label_a = keys[i]
            label_b = keys[j]
            centre_a = centres[label_a]
            centre_b = centres[label_b]
            distance = np.linalg.norm(centre_a - centre_b)

            # Create a dummy ax for symbols
            a_legend_key = ax.scatter([], [], marker='X', s=150, edgecolors='gray', color=colour_idx[label_a])
            arrow_key = ax.scatter([], [], marker=r"$\rightarrow$", color='gray')  # Joining symbol
            b_legend_key = ax.scatter([], [], marker='X', s=150, edgecolors='gray', color=colour_idx[label_b])

            legend_handles.append((a_legend_key, arrow_key, b_legend_key))  # Symbols

            # Add distance as label
            distance = f"{distance:.2f}"
            legend_labels.append(distance)  # Distance

    # Add ellipse areas to legend
    dummy, = ax.plot([], [], ' ')
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nEllipse Areas:")
    keys = list(shapely_ellipses.keys())
    for i in range(len(keys)):
        label_a = keys[i]
        area = shapely_ellipses[label_a]['area']

        legend_handles.append(ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_a]))  # Symbols

        # Add area as label
        area = f"{area:.2f}"
        legend_labels.append(area)

    # Add ellipse overlap areas to legend
    dummy, = ax.plot([], [], ' ')
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nEllipse Overlap Areas:")
    keys = list(shapely_ellipses.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):  # Only consider higher indices as don't need both directions
            label_a = keys[i]
            label_b = keys[j]
            overlap_area = compute_ellipse_overlap(shapely_ellipses[label_a]['ellipse'], shapely_ellipses[label_b]['ellipse'])

            small_area = min(shapely_ellipses[label_a]['area'], shapely_ellipses[label_b]['area'])
            percent = (overlap_area / small_area) * 100

            # Create a dummy ax for symbols
            a_legend_key = ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_a])
            arrow_key = ax.scatter([], [], marker=r"$\rightarrow$", color='gray')
            b_legend_key = ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_b])

            legend_handles.append((a_legend_key, arrow_key, b_legend_key))  # Symbols

            # Add area as label
            overlap_area = f"{overlap_area:.2f}"
            legend_labels.append(f"{overlap_area} ({percent:.2f}%)")

    # Plot centre of masses - separate loop so crosses are plotted on top of points
    for label, centre in centres.items():
        zorder = 0 if annotate else 5
        alpha = 0.3 if annotate else 1
        alpha = 0.3
        crosses = ax.scatter(centres[label][0], centres[label][1], alpha=alpha, marker='X', s=150, edgecolors='black', color=colour_idx[label], zorder=zorder)

    # Obtain robot IDs for annotating and identifying points on click
    if all_robot_ids is not None:
        plotted_coor = []
        plotted_ids = []
        # Collect plotted coordinates and IDs, add text to plot
        for plotted_idx in plot_idxs:
            coor = transformed_data[plotted_idx]
            rob_id = all_robot_ids[plotted_idx]
            # Label points
            if annotate:
                lower_labels = [42502, 252961]
                right_labels = [112648]
                vertical = 'bottom' if rob_id not in lower_labels else 'top'  # Bottom is above, top is below
                horizontal = 'right' if rob_id not in right_labels else 'left'  # Right is left, left is right
                ax.text(coor[0], coor[1], rob_id, fontsize=12, verticalalignment=vertical, horizontalalignment=horizontal, zorder=5)

            # Track plotted for annotating
            plotted_coor.append(coor)
            plotted_ids.append(rob_id)
        callback = partial(on_click, coordinates=plotted_coor, labels=plotted_ids)
        fig.canvas.mpl_connect("button_press_event", callback)

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
    :param dataset_labels: Labels list corresponding to transformed_data
    :return: Centre of mass dictionary with dataset label as key
    """
    unique_labels = sorted(set(dataset_labels))  # Alphabetical and consistent mapping
    centres = {}

    for label in unique_labels:
        idxs = np.array(dataset_labels) == label  # Bool array
        centres[label] = latent_2d_vectors[idxs].mean(axis=0)

    return centres


def create_shapely_ellipse(centre: np.ndarray, radii: np.ndarray, angle: np.float64) -> tuple[shapely.geometry.polygon.Polygon, float]:
    """
    Create a Shapely ellipse.
    For geometric calculations (overlap area).

    :param centre: Ellipse centre coordinates
    :param radii: Major and minor axes lengths
    :param angle: Angle in degrees
    :return: Shapely ellipse, area
    """
    # Circle with radius 1 around centre
    ellipse = Point(centre).buffer(1.0)

    # Scale along x and y
    ellipse = scale(ellipse, radii[0], radii[1])

    # Rotate clockwise to correct position. Shapely by default rotates counterclockwise
    ellipse = rotate(ellipse, -angle)

    return ellipse, ellipse.area


def evaluate_latent_vectors(latent_vectors: np.ndarray, dataset_labels: list[str], robot_ids: list = None, title: str = None, x_ax_min: int = 0, y_ax_min: int = 0, plot_set_colour="all", plot_idx: list = None, annotate: bool = False):
    # Normalise
    norm_latents, _ = normalise_latent(latent_vectors)

    # Apply PCA
    if np.all(norm_latents == 0):
        print(f"Normalised vector contains all zeros! PCA and UMAP plotting not possible for {title}.")
        return
    else:
        pca, pca_data = train_pca(norm_latents)

    # Apply UMAP
    # Global
    _, umap_data = train_umap(norm_latents)

    # pca_grean = np.where((pca_data[:, 0] < -3.5) & (pca_data[:, 1] > 2.0))[0]  # decrease x until found
    # print(f"\npca_green: {pca_grean}")
    # pca_blue = np.where((pca_data[:, 1] < -3.8))[0]  # decrease until found
    # print(f"\npca_blue: {pca_blue}")
    # umap_green = np.where((umap_data[:, 0] > 5.5) & (umap_data[:, 1] > 8.5))[0]
    # print(f"\numap green: {umap_green}")
    # umap_blue = np.where((umap_data[:, 0] > 5.0) & (umap_data[:, 1] < 2.5))[0]
    # print(f"\numap_blue: {umap_blue}")
    # return pca_green, umap_grean
    # exit()
    # Compute centres of mass
    pca_centres = compute_centres(pca_data, dataset_labels)
    umap_centres = compute_centres(umap_data, dataset_labels)

    if title is None:
        title_suffix = ""
    else:
        title_suffix = f" - {title}"

    # Plot
    plot_latent_space_evaluation(pca_data, dataset_labels, pca_centres, f"PCA Projection{title_suffix}", pca_model=pca, x_ax_min=x_ax_min, y_ax_min=y_ax_min, plot_set_colour=plot_set_colour, plot_idxs=plot_idx, all_robot_ids=robot_ids, annotate=annotate)
    plot_latent_space_evaluation(umap_data, dataset_labels, umap_centres, f"UMAP Projection{title_suffix}", x_ax_min=x_ax_min, y_ax_min=y_ax_min, plot_set_colour=plot_set_colour, plot_idxs=plot_idx, all_robot_ids=robot_ids, annotate=annotate)


def plot_latent_features(mean_vectors: np.ndarray, var_vectors: np.ndarray, robot_ids: list[int], dataset_labels: list[str], title: str = None, x_ax_min: int = 0, y_ax_min: int = 0, plot_set_colour="all", filename: str = None):
    unique_labels = sorted(set(dataset_labels))  # Ensures consistent legend and colours

    # Custom colours for dataset colours
    custom_colours = ['#2ca02c', '#17becf', '#9467bd']  # Green, Blue, Purple
    # Create colourmap
    cmap = ListedColormap(custom_colours)  # Dataset ellipse colours
    # Custom colourmap with 9 distinct colours (that does not contain gray)
    robot_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#bcbd22",  # Yellow
        "#17becf",  # Light blue
    ]
    colour_idx = {}

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(6, 6)
    ax = fig.add_subplot(gs[:, :-1])  # Main plot
    legend_ax = fig.add_subplot(gs[:, -1])  # Legend subplot

    legend_handles = []  # Symbols
    legend_labels = []  # Text labels

    # Plot points
    shapely_ellipses = {}
    for i, label in enumerate(unique_labels):
        # Set opacity and colour
        if plot_set_colour.lower() == label.lower():
            alpha = 1.0
            colour_idx[label] = cmap(i)
        elif plot_set_colour.lower() == "all":
            alpha = 0.7
            colour_idx[label] = cmap(i)
        else:
            alpha = 0.5
            colour_idx[label] = "gray"

        # For each dataset add a subtitle in legend
        dummy = ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label])
        legend_handles.append(dummy)  # For correct spacing
        display_lab = f"{' '.join(word.capitalize() for word in label.split('_'))}"
        legend_labels.append(display_lab)

        idxs = np.array(dataset_labels) == label  # Gets index based on original labels, order matches transformed_data

        # Get dataset vectors and robot IDs
        ds_mean_vec = mean_vectors[idxs, :][:3]
        ds_var_vec = var_vectors[idxs, :][:3]
        ds_ids = np.array(robot_ids)[idxs][:3]

        # Plot each robot sample points
        for j, (mean, var, rob_id) in enumerate(zip(ds_mean_vec, ds_var_vec, ds_ids)):
            display_label = f"Robot ID: {rob_id}"
            if colour_idx[label] == "gray":
                colour = colour_idx[label]  # Set to gray
                zorder = 1
            else:
                colour = robot_colors[(i * 3) + j]  # Select different colour for each robot
                zorder = 3

            # Scatter plot for points
            dots = ax.scatter(mean, var, marker="s", label=display_label, alpha=alpha, color=colour, linewidth=0.5, zorder=zorder)
            legend_handles.append(dots)
            legend_labels.append(display_label)

        ds_points = np.column_stack((ds_mean_vec.flatten(), ds_var_vec.flatten()))

        # Plot ellipse for dataset
        centre, radii, rotation = fit_ellipse(ax, ds_points, colour_idx[label], alpha=0.5)

        # Get shapely ellipse for overlaps and compute area, store for overlap calculation later
        ellipse, area = create_shapely_ellipse(centre, radii, rotation)
        shapely_ellipses[label] = {
            "colour": colour_idx[label],
            "area": area,
            "ellipse": ellipse
        }

    # Add ellipse areas to legend
    dummy, = ax.plot([], [], ' ')
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nEllipse Areas:")
    keys = list(shapely_ellipses.keys())
    for i in range(len(keys)):
        label_a = keys[i]
        area = shapely_ellipses[label_a]['area']

        legend_handles.append(
            ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_a]))  # Symbols

        # Add area as label
        area = f"{area:.2f}"
        legend_labels.append(area)

    # Add ellipse overlap areas to legend
    dummy, = ax.plot([], [], ' ')
    legend_handles.append(dummy)  # For correct spacing
    legend_labels.append("\nEllipse Overlap Areas:")
    keys = list(shapely_ellipses.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):  # Only consider higher indices as don't need both directions
            label_a = keys[i]
            label_b = keys[j]
            overlap_area = compute_ellipse_overlap(shapely_ellipses[label_a]['ellipse'],
                                                   shapely_ellipses[label_b]['ellipse'])

            small_area = min(shapely_ellipses[label_a]['area'], shapely_ellipses[label_b]['area'])
            percent = (overlap_area / small_area) * 100

            # Create a dummy ax for symbols
            a_legend_key = ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_a])
            arrow_key = ax.scatter([], [], marker=r"$\rightarrow$", color='gray')
            b_legend_key = ax.scatter([], [], marker='o', s=150, edgecolors='gray', color=colour_idx[label_b])

            legend_handles.append((a_legend_key, arrow_key, b_legend_key))  # Symbols

            # Add area as label
            overlap_area = f"{overlap_area:.2f}"
            legend_labels.append(f"{overlap_area} ({percent:.2f}%)")

    title = f"Latent Features - {title}"
    fig.suptitle(title)
    ax.set_xlabel("Mean Features")
    ax.set_ylabel("Log Variance Features")

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


def plot_feature_heatmap(mean_vectors: np.ndarray, var_vectors: np.ndarray, dataset_labels: list[str],
                         title: str = None, filename: str = None, latent_dim=16):
    unique_labels = sorted(set(dataset_labels))  # Ensures consistent dataset order

    mean_data_sets = {}  # X-axis (Mean Features)
    var_data_sets = {}   # Y-axis (Log Variance Features)

    vec_mins, vec_maxs = [], []

    # Collect data for each dataset
    for label in unique_labels:
        idxs = np.array(dataset_labels) == label  # Find matching dataset samples
        mean_data_sets[label] = mean_vectors[idxs, :]
        var_data_sets[label] = var_vectors[idxs, :]

        vec_mins.append(np.min(mean_data_sets[label]))
        vec_maxs.append(np.max(mean_data_sets[label]))
        vec_mins.append(np.min(var_data_sets[label]))
        vec_maxs.append(np.max(var_data_sets[label]))

    # Global min and max for consistent y-axes
    y_global_min, y_global_max = min(vec_mins), max(vec_maxs)

    # Set number of bins for y-axes
    y_bin_width = 0.1
    num_bins = int(np.ceil((y_global_max - y_global_min)/y_bin_width) + (2 / y_bin_width))  # Add buffer

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, len(unique_labels) * 2 + 1)

    processed_ds = {}
    all_counts = []

    # Find global min/max counts for shared colourbar, and prep data for plotting
    for col_idx, label in enumerate(unique_labels):
        # Set x-axis values and repeat for number of samples
        x_single_sample = np.array(range(latent_dim))
        x = np.tile(x_single_sample, len(mean_data_sets[label]))

        # Flatten datasets [y1_1..y1_latent, y2_1..y2_latent..., yn_1..yn_latent]
        mean_ds_flat = (mean_data_sets[label]).flatten()
        var_ds_flat = (var_data_sets[label]).flatten()

        # Store processed datasets
        processed_ds[label] = {
            'mean_ds': mean_ds_flat,
            'var_ds': var_ds_flat,
            'x': x
        }

        # Compute histogram without plotting
        hist_mean, _, _ = np.histogram2d(x, mean_ds_flat, bins=[latent_dim, num_bins], range=[[-1, latent_dim + 1], [y_global_min - 1, y_global_max + 1]])  # Range sets axes min and max
        hist_var, _, _ = np.histogram2d(x, var_ds_flat, bins=[latent_dim, num_bins], range=[[-1, latent_dim + 1], [y_global_min - 1, y_global_max + 1]])

        # Store histogram counts
        all_counts.append(hist_mean)
        all_counts.append(hist_var)

    # Find global max for consistent color scale
    max_count = np.max(all_counts)

    colourbar_mesh = None

    # Create a custom colourmap with lowest value as white
    cmap = LinearSegmentedColormap.from_list("", ["white", "darkblue"])

    # Iterate over datasets for plotting
    for col_idx, label in enumerate(unique_labels):
        # Each subplot takes up 2 grid spec columns and rows, each dataset is a new plot
        mean_vec_ax = fig.add_subplot(gs[:2, col_idx * 2:(col_idx + 1) * 2])
        var_vec_ax = fig.add_subplot(gs[2:, col_idx * 2:(col_idx + 1) * 2])

        # Plot datasets heatmap
        hist_mean = mean_vec_ax.hist2d(processed_ds[label]['x'], processed_ds[label]['mean_ds'], bins=[latent_dim, num_bins], range=[[-1, latent_dim + 1], [y_global_min - 1, y_global_max + 1]], cmap=cmap, vmin=0, vmax=max_count)  # type:ignore  # vmin and vmax (value) used for consistent colourbar
        hist_var = var_vec_ax.hist2d(processed_ds[label]['x'], processed_ds[label]['var_ds'], bins=[latent_dim, num_bins], range=[[-1, latent_dim + 1], [y_global_min - 1, y_global_max + 1]], cmap=cmap, vmin=0, vmax=max_count)  # type:ignore

        # Store first color mesh for colorbar
        if colourbar_mesh is None:
            colourbar_mesh = hist_mean[3]

        # Set subplot titles
        mean_vec_ax.set_title(f"{' '.join(word.capitalize() for word in label.split('_'))}")

    # Add a single colourbar for all heatmaps
    colourbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(colourbar_mesh, cax=colourbar_ax, label="Frequency")

    # Set global x-axis label
    fig.text(0.5, 0.05, "Features", ha='center', fontsize=14)

    # Set row titles
    fig.text(0.02, 0.7, "Mean Features", va='center', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.3, "Log Variance Features", va='center', rotation='vertical', fontsize=14, fontweight='bold')

    # Set main figure title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.show()

    # Save plot
    if filename is None:
        filename = title.lower().replace(" ", "_").replace("_-_", "-").replace(":", "")
    filepath = Path("PLOT_DIR") / f"{filename}.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    plt.close(fig)
