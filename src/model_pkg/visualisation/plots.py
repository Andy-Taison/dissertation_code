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
    Gets metric data from TrainingHistory object and transforms if required (for 'beta * kl', 'total loss', and 'accuracy').

    :param history: TrainingHistory object to obtain data from
    :param metric: Data to obtain, options - 'beta_kl', 'total_loss', 'accuracy', 'recon', 'kl', 'beta', 'accuracy', 'recall_weighted_avg', 'precision_weighted_avg', 'f1_weighted_avg', 'lr', 'training_time'
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :return: Train data, Validation data
    """
    data_train = history.train.get(metric, []) if metrics_from in ['train_and_val', 'train'] else []
    data_val = history.val.get(metric, []) if metrics_from in ['train_and_val', 'val'] else []

    if metric == 'beta_kl':
        data_train = [beta * kl for beta, kl in zip(history.train.get('beta', []), history.train.get('kl', []))]
        data_val = [beta * kl for beta, kl in zip(history.val.get('beta', []), history.val.get('kl', []))]

    if metric == 'total_loss':
        data_train = [recon + beta * kl for recon, kl, beta in
                      zip(history.train.get('recon', []), history.train.get('kl', []), history.train.get('beta', []))]
        data_val = [recon + beta * kl for recon, kl, beta in
                    zip(history.val.get('recon', []), history.val.get('kl', []), history.val.get('beta', []))]

    if metric == 'accuracy':
        data_train = [acc * 100 for acc in data_train]
        data_val = [acc * 100 for acc in data_val]

    return data_train, data_val


def process_metrics(training_histories: list[TrainingHistory], metrics: tuple[str], labels: list[str],
                    metrics_from: str) -> tuple[dict, int, list[float]]:
    """
    Collect and process data from list of TrainingHistory objects to a formatted dictionary.
    Maximum number of epochs from the data collected and a sorted list of maximum y scale data is also returned.

    :param training_histories: List of TrainingHistory objects to collect and process data from
    :param metrics: List of metrics to obtain, options - 'beta_kl', 'total_loss', 'accuracy', 'recon', 'kl', 'beta', 'accuracy', 'recall_weighted_avg', 'precision_weighted_avg', 'f1_weighted_avg', 'lr', 'training_time'
    :param labels: List of labels to ID each TrainingHistory dataset, prefix to 'legend_label' in returned dictionary
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :return: Metrics to plot dictionary, max data epochs, sorted list of maximum y scale data values
    """
    metrics_to_plot = {"legend_label": [], "data": [], "mode": [], "metric": [], "history": []}
    epochs = 0
    y_scales = []

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
                    y_scales.append(max(data))

    return metrics_to_plot, epochs, sorted(y_scales)


def determine_scale_limits(sorted_y_scales: list[float], threshold: int = 10) -> list[float]:
    """
    Determine the upper limits for at most 2 scales based on sorted maximum y-scale values.
    When the ratio between the maximum and minimum values are not within the threshold,
    function splits at the midpoint checking the ratio between the nearest split values.
    Values are moved from one group to another and rechecked. Values in the lower half do
    not have the potential to be split.

    :param sorted_y_scales: List of sorted maximum y-scale data values
    :param threshold: Ratio threshold for splitting into different scales
    :return: List of upper limits for scales (1 or 2 values)
    """
    # Calculate overall ratio
    overall_ratio = sorted_y_scales[-1] / sorted_y_scales[0]

    # If all values are within the threshold ratio, use one scale
    if overall_ratio <= threshold:
        return [sorted_y_scales[-1]]

    # Otherwise, split into two scales
    midpoint = len(sorted_y_scales) // 2
    group1 = sorted_y_scales[:midpoint]
    group2 = sorted_y_scales[midpoint:]

    # Adjust groups
    while group2 and (group2[0] / group1[-1]) <= threshold:
        group1.append(group2.pop(0))

    # Return the max of each group
    return [group1[-1], group2[-1]]


def plot_metrics(training_histories: TrainingHistory | list[TrainingHistory], *metrics: str,
                 filename: str | Path = None,
                 history_labels: list[str] = None, title: str = None, xlabel: str = 'Epochs', y1_label: str = None,
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
    :param title: Title for plot, when none provided and there are multiple TrainingHistory objects, uses 'alt_history_filename' or 'model_name', when a single TrainingHistory, set to 'Training Metrics'
    :param xlabel: X axis label
    :param y1_label: Left Y axis label, when none provided, uses metric name along with colour used in plot
    :param y2_label: Right Y axis label, used when 2 axes are plotted, when none provided, uses metric name along with colour used in plot
    :param metrics_from: Get data from - 'train_and_val', 'train', or 'val'
    :param y_scale_threshold: Threshold for ratio difference that determines when a second axis is required
    :param figsize: Figure size
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

    # Get data and calculate scales required
    processed_metrics, epochs, y_scales = process_metrics(training_histories, metrics, history_labels, metrics_from)
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
        primary_axis = True if max(data) <= y_scales[0] else False

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
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, y_scales[0] * 1.1)  # Always start y-axis at 0 and extend vertically by 10%
    ax.set_ylabel(y1_label or ", ".join(primary_axis_labels))
    if ax2:
        ax2.set_ylim(0, y_scales[1] * 1.1)
        ax2.set_ylabel(y2_label or ", ".join(secondary_axis_labels))
    ax.grid(True, alpha=0.3)  # Make grid opaque

    xticks = [int(x) for x in range(1, epochs + 1, max((epochs // 7), 1))]  # Ensures a reasonable number of ticks
    ax.set_xticks(xticks)

    # Add legend to the legend subplot
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_entries, loc='upper left')

    # Save plot
    filename = title if filename is None else filename
    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Plot saved to '{filepath.name}.png'")

    plt.show()
    plt.close(fig)
