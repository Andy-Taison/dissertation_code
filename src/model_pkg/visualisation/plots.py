from ..model.history_checkpoint import TrainingHistory
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def plot_metrics(training_histories: TrainingHistory | list[TrainingHistory], *metrics: str, labels: list[str] = None, title: str = "Training Metrics", xlabel: str = 'Epochs', ylabel: str = None, metrics_from: str = 'train_and_val'):
    """
    Function to plot metrics vs epochs.

    Metric options to plot include:
    - 'beta_kl': Beta * KL divergence (scaled KL div) per epoch.
    - 'total_loss': Total loss (recon + beta * kl) per epoch.
    - 'recon': Average reconstruction loss per epoch, weighted by class imbalance.
    - 'kl': KL divergence averaged across batches.
    - 'beta': Beta applied to KL divergence per epoch.
    - 'accuracy': Accuracy averaged across batches per epoch.
    - 'recall_weighted_avg': Recall weighed by descriptor value support, averaged across batches per epoch.
    - 'precision_weighted_avg': Precision weighted by descriptor value support, averaged across batches per epoch.
    - 'f1_weighted_avg': F1 score weighted by descriptor value support, averaged across batches per epoch.
    - 'lr': Learning rate used by optimizer per epoch.

    :param training_histories: TrainingHistory object to plot metrics from, provide single object or list of multiple objects
    :param metrics: Specify each metric to plot as a positional string argument
    :param labels: Optional list of labels for the histories, used in legend
    :param title: Title of plot
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param metrics_from: Plot metrics from 'train', 'val', or 'train_and_val'
    :return:
    """
    # Check data to plot
    if metrics_from not in ['train_and_val', 'train', 'val']:
        raise ValueError("Invalid 'metrics_from' for data to plot. Choose from 'train', 'val', or 'train_and_val'.")

    # Ensure 'training_histories' is a list
    if not isinstance(training_histories, list):
        training_histories = [training_histories]

    # Generate labels if none provided for each 'training_history'
    if labels is None:
        labels = [history.alt_history_filename if history.alt_history_filename is not None else history.model_name for history in training_histories]
    elif len(training_histories) != len(labels):
        raise ValueError("Number of training histories provided does not match number of labels provided.")

    max_length = 0

    # Create plot
    plt.figure(figsize=(10, 6))
    for history, label in zip(training_histories, labels):
        for metric in metrics:
            if metrics_from in ['train_and_val', 'train'] and metric in history.train:
                metric_length = len(history.train[metric])
                plt.plot(list(range(1, metric_length + 1)),  # Converted to list so .plot can handle
                         history.train[metric],
                         label=f"{label} - {metric} (Train)")
                max_length = max(metric_length, max_length)
            if metrics_from in ['train_and_val', 'val'] and metric in history.val:
                metric_length = len(history.val[metric])
                plt.plot(list(range(1, len(history.val[metric]) + 1)),
                         history.val[metric],
                         linestyle='--',
                         label=f"{label} - {metric} (Validation)")
                max_length = max(metric_length, max_length)

            # Beta * KL
            if metric == 'beta_kl':
                if metrics_from in ['train_and_val', 'train'] and 'beta' in history.train and 'kl' in history.train:
                    beta_kl_train = [beta * kl for beta, kl in zip(history.train['beta'], history.train['kl'])]
                    plt.plot(list(range(1, len(beta_kl_train) + 1)),
                             beta_kl_train,
                             label=f"{label} - Beta * KL (Train)")
                if metrics_from in ['train_and_val', 'val'] and 'beta' in history.val and 'kl' in history.val:
                    beta_kl_val = [beta * kl for beta, kl in zip(history.val['beta'], history.val['kl'])]
                    plt.plot(list(range(1, len(beta_kl_val) + 1)),
                             beta_kl_val,
                             linestyle='--',
                             label=f"{label} - Beta * KL (Val)")

            #  Total loss
            if metric == 'total_loss':
                if metrics_from in ['train_and_val', 'train'] and 'recon' in history.train and 'kl' in history.train and 'beta' in history.train:
                    total_loss_train = [recon + beta * kl for recon, kl, beta in zip(history.train['recon'], history.train['kl'], history.train['beta'])]
                    plt.plot(list(range(1, len(total_loss_train) + 1)),
                             total_loss_train,
                             label=f"{label} - Total Loss (Train)")
                if metrics_from in ['train_and_val', 'val'] and 'recon' in history.val and 'kl' in history.val and 'beta' in history.val:
                    total_loss_val = [recon + beta * kl for recon, kl, beta in zip(history.val['recon'], history.val['kl'], history.val['beta'])]
                    plt.plot(list(range(1, len(total_loss_val) + 1)),
                             total_loss_val,
                             linestyle='--',
                             label=f"{label} - Total Loss (Val)")

    plt.ylim(bottom=0)  # 0 always visible on the y-axis
    plt.xticks(list(range(1, max_length + 1)))  # Whole number epochs
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else ", ".join(metrics))
    plt.legend()
    plt.grid(True)
    plt.show()
