import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_aspect_metrics_for_epoch(metrics_by_class, epoch, output_dir='analysis_output'):
    aspects = list(metrics_by_class.keys())
    if not aspects:
        print(f"Warning: No aspects found in metrics_by_class for epoch {epoch}. Skipping plot.")
        return

    metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    metrics_labels = ['ACC', 'PPV', 'TPR', 'TNR', 'F1']

    data_matrix = np.zeros((len(aspects), len(metrics_names)))
    for i, aspect in enumerate(aspects):
        if not isinstance(metrics_by_class.get(aspect), dict):
            print(f"Warning: Metrics for aspect '{aspect}' in epoch {epoch} is not a dictionary. Skipping this aspect.")
            data_matrix[i, :] = np.nan  # Or some other placeholder
            continue
        for j, metric_name in enumerate(metrics_names):
            data_matrix[i, j] = metrics_by_class[aspect].get(metric_name, np.nan)

    plt.style.use('default')
    plt.figure(figsize=(15, 12))

    sns.heatmap(data_matrix,
                xticklabels=metrics_labels,
                yticklabels=aspects,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Score'},
                annot_kws={"size": 8})  # Smaller annotations if crowded

    plt.title(f'Performance Metrics by Aspect Category (Epoch {epoch})', pad=20, fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Aspect Categories', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=1.5)  # Add some padding

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'aspect_metrics_epoch_{epoch}.png')

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close()


def plot_training_and_f1_curves(train_losses, val_losses, f1_scores, output_dir='analysis_output'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 6))  # Adjusted figure size

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--')
    plt.title('Model Loss During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plotting F1 Score
    plt.subplot(1, 2, 2)
    if f1_scores and len(f1_scores) == len(epochs):  # Ensure f1_scores is not empty and matches epoch length
        plt.plot(epochs, f1_scores, label='Average F1 Score', color='green', marker='s', linestyle='-')
    else:
        print("Warning: F1 scores data is missing or length mismatch, F1 plot will be empty.")
    plt.title('Model Average F1 Score During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(pad=2.0)  # Increased padding

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'training_and_f1_curves.png')

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close()


def plot_individual_metric_trends(metrics_history, output_dir='analysis_output'):
    if not metrics_history:
        print("Warning: metrics_history is empty. Skipping individual metric trend plots.")
        return

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    num_epochs = len(metrics_history)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric_name in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        values = []

        for epoch_idx in range(num_epochs):
            epoch_data = metrics_history[epoch_idx]
            if not epoch_data or not isinstance(epoch_data, dict):
                print(f"Warning: Invalid or empty data for epoch {epoch_idx + 1} in metric '{metric_name}'. Appending NaN.")
                values.append(np.nan)
                continue

            current_epoch_metric_values = []
            for aspect_metrics in epoch_data.values():
                if isinstance(aspect_metrics, dict) and metric_name in aspect_metrics:
                    current_epoch_metric_values.append(aspect_metrics[metric_name])
                else:
                    current_epoch_metric_values.append(np.nan)  # Aspect might be missing or metric missing for aspect

            if not current_epoch_metric_values:
                values.append(np.nan)
            else:
                # Use np.nanmean to average, ignoring NaNs if some aspects don't have the metric
                mean_val = np.nanmean([v for v in current_epoch_metric_values if isinstance(v, (int, float))])
                values.append(mean_val)

        epochs_axis = range(1, num_epochs + 1)

        # Check if all values are NaN (e.g. metric was never recorded)
        if all(np.isnan(v) for v in values):
            print(f"Warning: All values for metric '{metric_name}' are NaN. Skipping plot for this metric.")
            plt.close()  # Close the empty figure
            continue

        plt.plot(epochs_axis, values, marker='o', linestyle='-')
        plt.title(f'Average {metric_name.capitalize()} Over Training Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name.capitalize(), fontsize=12)
        plt.xticks(ticks=epochs_axis)  # Ensure all epoch numbers are shown if space allows
        plt.grid(True, linestyle='--', alpha=0.7)

        output_path = os.path.join(output_dir, f'average_{metric_name}_trend.png')
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close()
