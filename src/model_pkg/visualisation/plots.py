"""
Functions to plot history metrics.
"""

from ..model.history_checkpoint import TrainingHistory
import matplotlib.pyplot as plt
from pathlib import Path
from ..config import PLOT_DIR

plt.style.use('dark_background')

def format_metric(metric: str) -> str:
    """
    Formats a metric for display

    :param metric: Metric to format
    :return: Formatted metric
    """
    if metric == "beta_kl":
        return "Beta * KL"
    if metric == "accuracy":
        return "Accuracy %"
    if len(metric) == 2:
        return metric.upper()
    return ' '.join(word.capitalize() for word in metric.split('_'))


def calculate_metric_data(history: TrainingHistory, metric: str, metrics_from: str) -> tuple[list[float], list[float]]:
    """
    Obtains metric data from TrainingHistory object and transforms if required (for 'beta * kl', 'total loss', and 'accuracy').

    :param history: TrainingHistory object to obtain data from
    :param metric: Data to obtain, options - 'beta_kl', 'total_loss', 'accuracy', 'recon', 'kl', 'beta', 'accuracy', 'recall_weighted_avg', 'precision_weighted_avg', 'f1_weighted_avg', 'lr', 'training_time'
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :return: Train data, Validation data
    """
    data_train = history.train.get(metric, []) if metrics_from in ['train_and_val', 'train'] else []
    data_val = history.val.get(metric, []) if metrics_from in ['train_and_val', 'val'] else []

    if metric == 'beta_kl':
        data_train = [beta * kl for beta, kl in zip(history.train.get('beta', []), history.train.get('kl', []))] if metrics_from in ['train_and_val', 'train'] else []
        data_val = [beta * kl for beta, kl in zip(history.val.get('beta', []), history.val.get('kl', []))] if metrics_from in ['train_and_val', 'val'] else []

    if metric == 'total_loss':
        data_train = [recon + beta * kl for recon, kl, beta in
                      zip(history.train.get('recon', []), history.train.get('kl', []), history.train.get('beta', []))] if metrics_from in ['train_and_val', 'train'] else []
        data_val = [recon + beta * kl for recon, kl, beta in
                    zip(history.val.get('recon', []), history.val.get('kl', []), history.val.get('beta', []))] if metrics_from in ['train_and_val', 'val'] else []

    if metric == 'accuracy':
        data_train = [acc * 100 for acc in data_train] if metrics_from in ['train_and_val', 'train'] else []
        data_val = [acc * 100 for acc in data_val] if metrics_from in ['train_and_val', 'val'] else []

    return data_train, data_val


def process_metrics(training_histories: list[TrainingHistory], metrics: tuple[str, ...], labels: list[str],
                    metrics_from: str, group_by_history: bool = False) -> tuple[list[dict], dict, dict] | tuple[dict, int, dict, list[float]]:
    """
    Collect and process data from list of TrainingHistory objects
    When 'group_by_history' is True, processes to a list of dictionaries for each TrainingHistory object.
    When 'group_by_history' is False, processes to a formatted dictionary.
    Maximum number of epochs from the data collected and a sorted list of maximum y scale data is also returned when 'group_by_history' is False.

    :param training_histories: List of TrainingHistory objects to collect and process data from
    :param metrics: List of metrics to obtain, options - 'beta_kl', 'total_loss', 'accuracy', 'recon', 'kl', 'beta', 'accuracy', 'recall_weighted_avg', 'precision_weighted_avg', 'f1_weighted_avg', 'lr', 'training_time'
    :param labels: List of labels to ID each TrainingHistory dataset, prefix to 'legend_label' in returned dictionary
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :param group_by_history: When True, organises data by history, when False, organises history by metric
    :return:
        - List of processed data, data min, and data max dictionaries when 'group_by_history',
        - Else metrics to plot dictionary, max data epochs, maximum y scale values dict by metric, sorted list of maximum y scale data values
    """
    if group_by_history:
        processed_data = []
        data_min = {}
        data_max = {}
        for history_counter, (history, label) in enumerate(zip(training_histories, labels)):
            history_data = {"label": label, "train": {}, "val": {}}
            for metric in metrics:
                train_data, val_data = calculate_metric_data(history, metric, metrics_from)
                if train_data:
                    history_data["train"][metric] = train_data
                    if metric not in data_min or min(train_data) < data_min[metric]:
                        data_min[metric] = min(train_data)
                    if metric not in data_max or max(train_data) > data_max[metric]:
                        data_max[metric] = max(train_data)
                if val_data:
                    history_data["val"][metric] = val_data
                    if metric not in data_min or min(val_data) < data_min[metric]:
                        data_min[metric] = min(val_data)
                    if metric not in data_max or max(val_data) > data_max[metric]:
                        data_max[metric] = max(val_data)
            processed_data.append(history_data), data_min, data_max
        return processed_data, data_min, data_max
    else:
        metrics_to_plot = {"legend_label": [], "data": [], "mode": [], "metric": [], "history": []}
        epochs = 0
        metric_max_y = {}  # Used to determine which scale to plot metric on when twin scales

        for history_counter, (history, label) in enumerate(zip(training_histories, labels)):
            for metric in metrics:
                data_train, data_val = calculate_metric_data(history, metric, metrics_from)
                if not data_train + data_val:
                    continue

                for data, mode in [(data_train, "train"), (data_val, "val")]:
                    if data:
                        formatted_metric = format_metric(metric)
                        metrics_to_plot['legend_label'].append(f"{label}{formatted_metric} ({mode.capitalize()})")
                        metrics_to_plot['data'].append(data)
                        metrics_to_plot['mode'].append(mode)
                        metrics_to_plot['metric'].append(formatted_metric)
                        metrics_to_plot['history'].append(history_counter)
                        epochs = max(epochs, len(data))
                        if formatted_metric not in metric_max_y or max(data) > metric_max_y[formatted_metric]:
                            metric_max_y[formatted_metric] = max(data)

        y_scales = list(metric_max_y.values())  # Used to determine twin scale

        return metrics_to_plot, epochs, metric_max_y, sorted(y_scales)


def determine_scale_limits(sorted_y_scales: list[float], threshold: int = 10) -> list[float]:
    """
    Determine the upper limits for at most 2 scales based on sorted maximum y-scale values.
    When the ratio between the maximum and minimum values are not within the threshold,
    function splits at the midpoint (in the hope to create even groupings) checking the ratio
    between the nearest split values. Values are moved from one group to another and rechecked.
    Starts at the beginning of the list if starting at midpoint did not find the threshold split.
    When no ideal split between two adjacent elements, returns 1 element (for a single scale).

    :param sorted_y_scales: List of sorted maximum y-scale data values
    :param threshold: Ratio threshold for splitting into different scales
    :return: List of upper limits for scales (1 or 2 values)
    """
    # Calculate overall ratio
    overall_ratio = sorted_y_scales[-1] / sorted_y_scales[0]

    # If all values are within the threshold ratio, use one scale
    if overall_ratio <= threshold or len(sorted_y_scales) == 1:
        return [sorted_y_scales[-1]]

    # Otherwise, split at the middle (hopefully to end up with equally split data if multiple potential ratio split points)
    midpoint = len(sorted_y_scales) // 2
    group1 = sorted_y_scales[:midpoint]
    group2 = sorted_y_scales[midpoint:]

    no_ideal_split = False
    # Adjust groups
    while (group2[0] / group1[-1]) <= threshold:
        group1.append(group2.pop(0))
        if not group2:
            if no_ideal_split:
                # Use one scale
                return [group1[-1]]
            # If got to the end of the list, look for split in beginning of list, one more iteration only
            group1 = [sorted_y_scales[0]]
            group2 = sorted_y_scales[1:]
            no_ideal_split = True

    # Return the max of each group
    return [group1[-1], group2[-1]]


def check_inputs(training_histories: TrainingHistory | list[TrainingHistory], history_labels: list[str], title: str, metrics_from: str) -> tuple[list[TrainingHistory], list[str], str]:
    """
    Helper function to check and set inputs.

    :param training_histories: TrainingHistory objects to put in a list if not already
    :param history_labels: Labels used to ID TrainingHistory objects, when None, labels are generated based on training_histories
    :param title: Sets a title if None based on training_histories passed
    :param metrics_from: Checks where to obtain data is valid
    :return: training_histories, history_labels, title
    """
    # Check data to plot
    if metrics_from not in ['train_and_val', 'train', 'val']:
        raise ValueError("Invalid 'metrics_from'. Choose from 'train', 'val', or 'train_and_val'.")

    # Ensure 'training_histories' is a list
    training_histories = training_histories if isinstance(training_histories, list) else [training_histories]

    # Set title
    if title is None:
        if len(training_histories) > 1:
            title = "Training Metrics"
        else:
            title = training_histories[0].alt_history_filename or training_histories[0].model_name

    # Generate and check history_labels
    if not history_labels:
        if len(training_histories) > 1:
            history_labels = []
            for history in training_histories:
                history_labels.append(
                    f"{history.alt_history_filename} - " if history.alt_history_filename else f"{history.model_name} - ")
        else:
            history_labels = [""]

    if len(training_histories) != len(history_labels):
        raise ValueError("Number of 'training_histories' and 'history_labels' must match.")

    return training_histories, history_labels, title


def plot_metrics_vs_epochs(training_histories: TrainingHistory | list[TrainingHistory], *metrics: str,
                           filename: str = None,
                           history_labels: list[str] = None, title: str = None, y1_label: str = None,
                           y2_label: str = None, metrics_from: str = 'train_and_val', y_scale_threshold: int = 10,
                           figsize: tuple[float, float] = (15, 6)):
    """
    Function to plot different metrics from multiple TrainingHistory objects in a line plot.
    Supports up to 5 different TrainingHistory objects (2-5 are denoted using markers).
    Multiple metrics can be plotted, specify each as a positional string argument, each plotted in a different colour.
    Supports up to 6 different metrics.
    'y_scale_threshold' determines the ratio difference between metric dataset maximum values, that determines when a second y-axis is required.
    When a second axis is used, this will be plotted on the right.
    When no y labels provided, the metrics plotted against them are used as the label along with the colours used.
    Data from 'train' will be plotted as a solid line, data from 'val' will be plotted as a dotted line.
    Generated plot is stored in 'PLOT_DIR' as specified in config.

    :param training_histories: TrainingHistory or list of TrainingHistory objects to plot data from
    :param metrics: Each metric should be a positional argument, options - 'beta_kl', 'total_loss', 'accuracy', 'recon', 'kl', 'beta', 'accuracy', 'recall_weighted_avg', 'precision_weighted_avg', 'f1_weighted_avg', 'lr', 'training_time'
    :param filename: Filename to save generated plot. When none provided, the plot title is used. Stores in 'PLOT_DIR' as specified in config
    :param history_labels: List of labels to ID TrainingHistories, used in legend. When none provided and there are multiple TrainingHistory objects, uses 'alt_history_filename' or 'model_name'
    :param title: Title for plot, when none provided and there is a single TrainingHistory object, uses 'alt_history_filename' or 'model_name', when multiple TrainingHistory objects, set to 'Training Metrics'
    :param y1_label: Left Y axis label, when none provided, uses metric name along with colour used in plot
    :param y2_label: Right Y axis label, used when 2 axes are plotted, when none provided, uses metric name along with colour used in plot
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :param y_scale_threshold: Threshold for ratio difference that determines when a second axis is required
    :param figsize: Figure size
    """
    training_histories, history_labels, title = check_inputs(training_histories, history_labels, title, metrics_from)

    # Get data and calculate scales required
    processed_metrics, epochs, metric_max_y, y_scales = process_metrics(training_histories, metrics, history_labels, metrics_from)
    y_scales = determine_scale_limits(y_scales, y_scale_threshold)

    # Initialise plot
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 6)
    ax = fig.add_subplot(gs[:, :-1])  # Main plot
    legend_ax = fig.add_subplot(gs[:, -1])  # Legend subplot
    fig.suptitle(title)

    # Create additional axes if needed
    ax2 = ax.twinx() if len(y_scales) > 1 else None

    # Colour lists and trackers for each axis
    colour_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]
    colours_by_metric = {}
    colour_idx = 0
    primary_axis_labels = set()
    secondary_axis_labels = set()

    # Markers and styles
    markers = ["None", "v", "x", "D", "o"]
    linestyles = {"train": "solid", "val": "dotted"}

    # Legend entries
    legend_entries = []

    # Plot data
    for data, label, metric, mode, history_counter in zip(
            processed_metrics['data'],
            processed_metrics['legend_label'],
            processed_metrics['metric'],
            processed_metrics['mode'],
            processed_metrics['history']):
        primary_axis = True if metric_max_y[metric] <= y_scales[0] else False

        # Set colour for each metric
        if metric not in colours_by_metric:
            if primary_axis:
                colours_by_metric[metric] = colour_list[colour_idx]
                colour_idx += 1
            else:
                colours_by_metric[metric] = colour_list[colour_idx]
                colour_idx += 1

        # Select axis and set axis label
        if primary_axis:
            axis = ax
            primary_axis_labels.add(f"{metric} ({colours_by_metric[metric].split(':')[1].capitalize()})")
        else:
            axis = ax2
            secondary_axis_labels.add(f"{metric} ({colours_by_metric[metric].split(':')[1].capitalize()})")

        # Plot
        line, = axis.plot(
            list(range(1, epochs + 1)),
            data,
            label=label,
            color=colours_by_metric[metric],
            linestyle=linestyles[mode],
            marker=markers[history_counter],
            markersize=10
        )
        legend_entries.append(line)

    # Configure axes
    ax.set_xlabel("Epochs")
    ax_current_lim = ax.get_ylim()
    ax.set_ylim(0, ax_current_lim[1] * 1.1)  # Always start y-axis at 0 and extend vertically by 10%
    ax.set_ylabel(y1_label or ", ".join(primary_axis_labels))
    if ax2:
        ax2_current_lim = ax2.get_ylim()
        ax2.set_ylim(0, ax2_current_lim[1] * 1.1)
        ax2.set_ylabel(y2_label or ", ".join(secondary_axis_labels))
    ax.grid(True, alpha=0.3)  # Make grid opaque

    xticks = [int(x) for x in range(1, epochs + 1, max((epochs // 7), 1))]  # Ensures a reasonable number of ticks
    ax.set_xticks(xticks)

    # Add legend to the legend subplot
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_entries, loc='upper left')

    # Save plot
    filename = title.replace(" ", "_").lower() if filename is None else filename
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    plt.show()
    plt.close(fig)


def plot_loss_tradeoffs(training_histories: TrainingHistory | list[TrainingHistory], loss: str,
                        filename: str | Path = None, history_labels: list[str] = None, title: str = None,
                        metrics_from: str = 'train_and_val', figsize: tuple[float, float] = (15, 6),
                        x_lowerbound: float = 0, y_lowerbound: float = 0):
    """
    Function to plot 1 or more TrainingHistory object loss metrics.
    Loss options - 'kl' (KL divergence vs reconstruction loss), 'kl_beta' (Beta scaled KL divergence vs reconstruction loss), 'total_loss' (Total loss vs weighted F1 average).
    When multiple TrainingHistory objects provided, data points are coloured by TrainingHistory, else they are coloured by 'train' or 'val' metrics.
    Supports up to 5 TrainingHistory objects.

    :param training_histories: TrainingHistory or list of TrainingHistory objects to plot data from
    :param loss: 'kl' (KL divergence vs reconstruction loss), 'kl_beta' (Beta scaled KL divergence vs reconstruction loss), 'total_loss' (Total loss vs weighted F1 average)
    :param filename: Filename to save generated plot. When none provided, the plot title is used. Stores in 'PLOT_DIR' as specified in config
    :param history_labels: List of labels to ID TrainingHistories, used in legend. When none provided and there are multiple TrainingHistory objects, uses 'alt_history_filename' or 'model_name'
    :param title: Title for plot, when none provided and there is a single TrainingHistory object, uses 'alt_history_filename' or 'model_name', when multiple TrainingHistory objects, set based on 'loss'
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :param figsize: Figure size
    :param x_lowerbound: Adjust the x-axis lowerbound
    :param y_lowerbound: Adjust the y-axis lowerbound
    :return:
    """
    training_histories, history_labels, title = check_inputs(training_histories, history_labels, title, metrics_from)

    # Update title and set metrics to obtain
    match loss:
        case 'kl':
            if title == 'Training Metrics':
                title = "KL Divergence vs Reconstruction Loss"
            metrics = ('recon', 'kl')
            xlabel = "Reconstruction Loss"
            ylabel = "KL Divergence"
        case 'kl_beta':
            if title == 'Training Metrics':
                title = "Beta Scaled KL Divergence vs Reconstruction Loss"
            metrics = ('recon', 'beta_kl')
            xlabel = "Reconstruction Loss"
            ylabel = "Beta Scaled KL Divergence"
        case 'total_loss':
            if title == 'Training Metrics':
                title = "Total Loss vs Weighted F1 Average"
            metrics = ('f1_weighted_avg', 'total_loss')
            xlabel = "Weighted F1 Average"
            ylabel = "Total Loss"
        case _:
            raise ValueError("Invalid 'loss'. Choose from 'kl', 'kl_beta', or 'total_loss'.")

    # Get data and calculate scales required
    grouped_data, data_min, data_max = process_metrics(training_histories, metrics, history_labels, metrics_from, group_by_history=True)

    # Initialise plot
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 6)
    ax = fig.add_subplot(gs[:, :-1])  # Main plot
    legend_ax = fig.add_subplot(gs[:, -1])  # Legend subplot
    fig.suptitle(title)

    colour_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = {"train": ".", "val": "x"}

    # Legend entries
    legend_entries = []

    # Plot data
    for history_idx, history_data in enumerate(grouped_data):
        label = history_data["label"]
        for mrk_idx, (mode, marker) in enumerate(markers.items()):
            colour = colour_list[history_idx] if len(grouped_data) > 1 else colour_list[mrk_idx]  # Colour by history if multiple, else by mode
            x_data = history_data[mode].get(metrics[0])
            y_data = history_data[mode].get(metrics[1])
            if x_data and y_data:
                group = ax.scatter(x_data, y_data, label=f"{label}{mode.capitalize()}", color=colour, marker=marker)
                legend_entries.append(group)

    # Configure axes
    ax.set_xlim(x_lowerbound, data_max[metrics[0]] * 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lowerbound, data_max[metrics[1]] * 1.1)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)  # Make grid opaque

    # Add legend to the legend subplot
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_entries, loc='upper left')

    # Save plot
    filename = title.replace(" ", "_").lower() if filename is None else filename
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}'")

    plt.show()
    plt.close(fig)


def generate_plots(history: TrainingHistory, model_name: str):
    plot_metrics_vs_epochs(history, "recon", "beta_kl", filename=f"{model_name}/{model_name}_recon-beta_kl_vs_epochs")
    plot_metrics_vs_epochs(history, "f1_weighted_avg", filename=f"{model_name}/{model_name}_f1_vs_epochs")
    plot_metrics_vs_epochs(history, "total_loss", filename=f"{model_name}/{model_name}_total_loss_vs_epochs")
    plot_metrics_vs_epochs(history, "accuracy", filename=f"{model_name}/{model_name}_acc_vs_epochs")
    plot_metrics_vs_epochs(history, "coor_loss", "desc_loss", "pad_penalty", "collapse_penalty", filename=f"{model_name}/{model_name}_raw_recon_loss_parts_vs_epochs")
    plot_metrics_vs_epochs(history, "scaled_coor_loss", "scaled_desc_loss", "scaled_collapse_penalty", filename=f"{model_name}/{model_name}_scaled_recon_loss_parts_vs_epochs")

    plot_loss_tradeoffs(history, "kl", filename=f"{model_name}/{model_name}_kl_vs_recon")
    plot_loss_tradeoffs(history, "kl_beta", filename=f"{model_name}/{model_name}_beta_kl_vs_recon")
    plot_loss_tradeoffs(history, "total_loss", filename=f"{model_name}/{model_name}_total_loss_vs_f1")
