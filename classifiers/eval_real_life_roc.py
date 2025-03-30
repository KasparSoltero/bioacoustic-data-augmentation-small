# this file gets roc curve for a given model (using real life little owl dataset n~70)

import pickle
import os
import torch
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from classifiers.classifier_small import BirdSoundModel, TrainingParameters, DefaultAudio, FilePaths, real_life_evaluate_with_roc, test_inference

def calculate_roc_curves(experiment_dirs, experiment_caption_prefix, experiment_caption_vals):
    # Load audio configuration
    audio_cfg = DefaultAudio()
    # Load use case
    with open('classifiers/use_case.yaml') as f:
        use_case = yaml.safe_load(f)
    train_cfg = TrainingParameters(options=use_case)
    paths = FilePaths(use_case)

    # single consensus roc plot
    # model_path = 'classifiers/evaluation_results/Exp_12/Results/binary_classifier-epoch=14-val_auc=0.858.ckpt'
    # bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # model_state_dict = torch.load(model_path)
    # bird_model.load_state_dict(model_state_dict)

    # # here we load the model and evaluate it on the real life dataset
    # bird_model.eval()
    # model_path2 = 'classifiers/evaluation_results/Exp_14/Results/binary_classifier-epoch=15-val_auc=0.873.ckpt'
    # bird_model2 = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # model_state_dict2 = torch.load(model_path2)
    # bird_model2.load_state_dict(model_state_dict2)
    # bird_model2.eval()
    # bird_models = [bird_model, bird_model2]
    # print('Model loaded')
    # eval_dir = paths.EVAL_DIR
    # metrics = real_life_evaluate_with_roc(bird_models, eval_dir, audio_cfg, consensus=True)

    rocs = {}
    
    for experiment_dir, n_val in zip(experiment_dirs, experiment_caption_vals):
        results_dir = Path('classifiers/evaluation_results') / experiment_dir / 'Results'
        model_paths = [f for f in os.listdir(results_dir) if f.startswith('binary_classifier-epoch=') and f.endswith('.ckpt')]
        rocs[experiment_caption_prefix + str(n_val)] = []
        for model_path in model_paths:
            bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
            model_state_dict = torch.load(results_dir / model_path)
            bird_model.load_state_dict(model_state_dict)
            bird_model.eval()
            eval_dir = paths.EVAL_DIR
            metrics = real_life_evaluate_with_roc(bird_model, eval_dir, audio_cfg, plot=False)
            rocs[experiment_caption_prefix + str(n_val)].append(metrics['roc_data']) #fpr, tpr, thresholds
            print(f'Experiment {experiment_dir} model {model_path} metrics: {metrics}')

    # Save the ROC curves to a file
    save_roc_curves(rocs)

def save_roc_curves(rocs, filename='roc_curves.pkl'):
    """
    Save ROC curves to a pickle file.
    
    Args:
        rocs (dict): Dictionary containing ROC curve data
        filename (str): Path to save the file
    """
    with open(filename, 'wb') as f:
        pickle.dump(rocs, f)
    print(f"ROC curves saved to {filename}")

def load_roc_curves(filename):
    """
    Load ROC curves from a pickle file.
    
    Args:
        filename (str): Path to the saved file
        
    Returns:
        dict: Dictionary containing ROC curve data
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            rocs = pickle.load(f)
        print(f"ROC curves loaded from {filename}")
        return rocs
    else:
        print(f"File {filename} not found.")
        return None

def plot_all_roc_curves(rocs):
    plt.figure(figsize=(12, 10))
    for i, (experiment, roc_values) in enumerate(rocs.items()):
        color = colors[i % len(colors)]
        
        # Plot each ROC curve for this experiment
        for j, roc_data in enumerate(roc_values):
            # Handle different possible formats of roc_data
            if isinstance(roc_data, dict):
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
            else:  # Assume it's a tuple/list with fpr, tpr, thresholds
                fpr, tpr = roc_data[0], roc_data[1]
            
            # Use consistent color with some transparency for each experiment
            alpha = 0.7
            plt.plot(fpr, tpr, color=color, alpha=alpha, linewidth=1)
            
            # Only add to legend for the first curve in each experiment
            if j == 0:
                plt.plot([], [], color=color, label=experiment)
    
    # Add reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Training Set Sizes')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_roc_curves_with_fpr_bounds(rocs):
    plt.figure(figsize=(9, 8))
    # Common y-axis (TPR) points for interpolation
    common_tpr = np.linspace(0, 1, 1000)
    
    for i, (experiment, roc_values) in enumerate(rocs.items()):
        color = colors[i % len(colors)]
        
        # Arrays to store interpolated FPR values for all models in this experiment
        interpolated_fprs = []
        
        # Process each model's ROC curve
        for roc_data in roc_values:
            # Extract FPR and TPR
            if isinstance(roc_data, dict):
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
            else:
                fpr, tpr = roc_data[0], roc_data[1]
            
            # Convert to numpy arrays if they aren't already
            fpr = np.array(fpr)
            tpr = np.array(tpr)
            
            # Sort the points by TPR for proper interpolation
            # (ROC curves might not always be strictly increasing)
            sort_indices = np.argsort(tpr)
            tpr_sorted = tpr[sort_indices]
            fpr_sorted = fpr[sort_indices]
            
            # Interpolate FPR values at common TPR points
            # Limit interpolation to the range of TPR values actually present
            valid_tpr_mask = (common_tpr >= tpr_sorted[0]) & (common_tpr <= tpr_sorted[-1])
            interpolated_fpr = np.ones_like(common_tpr)
            
            # Only interpolate where we have valid TPR values
            interpolated_fpr[valid_tpr_mask] = np.interp(
                common_tpr[valid_tpr_mask], tpr_sorted, fpr_sorted
            )
            interpolated_fprs.append(interpolated_fpr)
        
        # Convert to numpy array for easier min/max calculation
        interpolated_fprs = np.array(interpolated_fprs)
        
        # Find min and max FPR values at each TPR point
        fpr_min = np.min(interpolated_fprs, axis=0)
        fpr_max = np.max(interpolated_fprs, axis=0)
        
        # Calculate mean FPR (for the central line)
        fpr_mean = np.mean(interpolated_fprs, axis=0)

        mean_auc = auc(fpr_mean, common_tpr)
        
        # Plot the mean line - but flip axes to maintain traditional ROC curve layout
        plt.plot(fpr_mean, common_tpr, color=color, 
                label=f'{experiment} (mean AUC: {mean_auc:.3f})', 
                linewidth=2)
        
        # Plot the filled area between min and max - with flipped axes
        plt.fill_betweenx(common_tpr, fpr_min, fpr_max, color=color, alpha=0.2)
    
    # Add reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Min/Max FPR Bounds for Different Training Set Sizes')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()


def plot_roc_curves_with_bounds(rocs, limit_curves=None, legend_title='Training Set Size'):
    if limit_curves is not None and isinstance(limit_curves, int):
        print(f"Limiting to top {limit_curves} curves per experiment")

    plt.figure(figsize=(9, 8))
    # Common x-axis points for interpolation
    common_fpr = np.linspace(0, 1, 1000)
    
    for i, (experiment, roc_values) in enumerate(rocs.items()):
        color = colors[i % len(colors)]
        
        # Arrays to store interpolated TPR values for all models in this experiment
        interpolated_tprs = []
        interpolated_data = [] #store auc values for limit_curves
        
        # Process each model's ROC curve
        for j, roc_data in enumerate(roc_values):
            # Extract FPR and TPR
            if isinstance(roc_data, dict):
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
            else:
                fpr, tpr = roc_data[0], roc_data[1]
            
            # Convert to numpy arrays if they aren't already
            fpr = np.array(fpr)
            tpr = np.array(tpr)

            # limit functionality
            sort_indices = np.argsort(fpr)
            fpr_sorted_auc = fpr[sort_indices]
            tpr_sorted_auc = tpr[sort_indices]
            individual_auc = auc(fpr_sorted_auc, tpr_sorted_auc)
            
            # Interpolate TPR values at common FPR points
            interpolated_tpr = np.interp(common_fpr, fpr, tpr)
            interpolated_tpr[0] = 0.0  # Force start at 0
            interpolated_tprs.append(interpolated_tpr)
            interpolated_data.append((individual_auc, interpolated_tpr))

        # --- Start: Limit functionality ---
        num_curves_processed = len(interpolated_data)
        if limit_curves is not None and isinstance(limit_curves, int) and limit_curves>0 and num_curves_processed>limit_curves:
            interpolated_data.sort(key=lambda x: x[0], reverse=True)
            selected_data = interpolated_data[:limit_curves]
        else:
            selected_data = interpolated_data

        # Handle case where no curves are left after filtering or initially
        if not selected_data:
             print(f"Warning: No ROC curves to plot for experiment {experiment}.")
             continue # Skip to the next experiment
        
        interpolated_tprs_for_avg = np.array([data[1] for data in selected_data])
        # --- End: Limit functionality ---
        
        # Find min and max TPR values at each FPR point
        tpr_min = np.min(interpolated_tprs_for_avg, axis=0)
        tpr_max = np.max(interpolated_tprs_for_avg, axis=0)
        
        # Calculate mean TPR (for the central line)
        tpr_mean = np.mean(interpolated_tprs_for_avg, axis=0)

        mean_auc = auc(common_fpr, tpr_mean)
        
        # Plot the mean line
        plt.plot(common_fpr, tpr_mean, color=color, 
                label=f'{experiment} ({mean_auc:.2f})',
                linewidth=2)
        
        # Plot the filled area between min and max
        plt.fill_between(common_fpr, tpr_min, tpr_max, color=color, alpha=0.2)
    
    # Add reference line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#666')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=12, title=legend_title)
    plt.grid(True)
    plt.tight_layout()


def plot_rocs_sidebyside(rocs1, rocs2, colors1, colors2, legend_title1, legend_title2, limit_curves=None):
    """
    Plots two sets of ROC curves with bounds side-by-side on a shared y-axis.

    Args:
        rocs1 (dict): Dictionary for the first set of ROC data. Keys are experiment names,
                      values are lists of ROC curve data (e.g., [(fpr, tpr), ...]).
        rocs2 (dict): Dictionary for the second set of ROC data.
        colors1 (list): List of colors for the first plot.
        colors2 (list): List of colors for the second plot.
        legend_title1 (str): Title for the legend of the first plot.
        legend_title2 (str): Title for the legend of the second plot.
        limit_curves (int, optional): Maximum number of curves (sorted by AUC) to use
                                     for calculating bounds per experiment. Defaults to None (use all).
    """

    if limit_curves is not None and isinstance(limit_curves, int):
        print(f"Limiting to top {limit_curves} curves per experiment for bounds calculation.")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True) # Create 1x2 subplots, share y-axis

    common_fpr = np.linspace(0, 1, 1000) # Common x-axis points for interpolation

    # --- Helper function to plot one set of ROCs ---
    def _plot_single_roc_set(ax, rocs, colors, legend_title):
        for i, (experiment, roc_values) in enumerate(rocs.items()):
            color = colors[i % len(colors)]

            # Arrays to store interpolated TPR values and AUCs
            interpolated_data = [] # Stores (auc, interpolated_tpr) tuples

            if not roc_values:
                print(f"Warning: No ROC curves found for experiment '{experiment}' in this set.")
                continue

            # Process each model's ROC curve within the experiment
            for j, roc_data in enumerate(roc_values):
                # Extract FPR and TPR
                if isinstance(roc_data, dict):
                    fpr = roc_data.get('fpr', [])
                    tpr = roc_data.get('tpr', [])
                elif isinstance(roc_data, (list, tuple)) and len(roc_data) >= 2:
                    fpr, tpr = roc_data[0], roc_data[1]
                else:
                    print(f"Warning: Skipping invalid roc_data format in experiment '{experiment}', item {j}")
                    continue

                fpr_orig = np.array(fpr) # Keep original arrays
                tpr_orig = np.array(tpr)
                if fpr_orig.size == 0 or tpr_orig.size == 0 or fpr_orig.size != tpr_orig.size:
                    print(f"Warning: Skipping empty or mismatched FPR/TPR in experiment '{experiment}', item {j}")
                    continue

                # --- AUC Calculation (using sorted data, as before) ---
                sort_indices = np.argsort(fpr_orig)
                fpr_sorted = fpr_orig[sort_indices]
                tpr_sorted = tpr_orig[sort_indices]

                if len(np.unique(fpr_sorted)) < 2:
                     print(f"Warning: Skipping curve with insufficient unique FPR values for AUC calculation in experiment '{experiment}', item {j}")
                     individual_auc = 0.0
                else:
                    individual_auc = auc(fpr_sorted, tpr_sorted) # AUC uses sorted values

                # --- Interpolation for Plotting (using ORIGINAL potentially unsorted data) ---
                # Use the original fpr_orig, tpr_orig arrays here to match the user's single plot function.
                # Note: This deviates from np.interp's requirement that the x-array (fpr_orig) be monotonic.
                interpolated_tpr = np.interp(common_fpr, fpr_orig, tpr_orig)
                # --- End Modification ---

                interpolated_tpr[0] = 0.0  # Force start at 0
                interpolated_data.append((individual_auc, interpolated_tpr)) # Store AUC (from sorted) and interpolated TPR (from original)

            # --- Start: Limit functionality ---
            num_curves_processed = len(interpolated_data)
            if limit_curves is not None and isinstance(limit_curves, int) and limit_curves > 0 and num_curves_processed > limit_curves:
                interpolated_data.sort(key=lambda x: x[0], reverse=True) # Sort by AUC descending
                selected_data = interpolated_data[:limit_curves]
            else:
                selected_data = interpolated_data

            # Handle case where no curves are left after filtering or initially
            if not selected_data:
                 print(f"Warning: No valid ROC curves left to plot for experiment '{experiment}' after processing/filtering.")
                 continue # Skip to the next experiment

            interpolated_tprs_for_avg = np.array([data[1] for data in selected_data])
             # --- End: Limit functionality ---


            if interpolated_tprs_for_avg.size == 0:
                 print(f"Warning: No interpolated TPRs to calculate bounds/mean for experiment '{experiment}'.")
                 continue

            # Find min and max TPR values at each FPR point
            tpr_min = np.min(interpolated_tprs_for_avg, axis=0)
            tpr_max = np.max(interpolated_tprs_for_avg, axis=0)

            # Calculate mean TPR (for the central line)
            tpr_mean = np.mean(interpolated_tprs_for_avg, axis=0)

            mean_auc = auc(common_fpr, tpr_mean)

            # Plot the mean line
            ax.plot(common_fpr, tpr_mean, color=color,
                    label=f'{experiment} ({mean_auc:.2f})',
                    linewidth=2)

            # Plot the filled area between min and max
            ax.fill_between(common_fpr, tpr_min, tpr_max, color=color, alpha=0.2)

        # Add reference line (random classifier)
        ax.plot([0, 1], [0, 1], linestyle='--', color='#666')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0]) # Ensure y-lim is consistent
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.legend(loc="lower right", fontsize=14, title=legend_title)
        ax.grid(True)

    # --- Plotting ---
    # Plot left side
    _plot_single_roc_set(axes[0], rocs1, colors1, legend_title1)
    axes[0].set_ylabel('True Positive Rate', fontsize=14) # Set Y label only on the left plot

    # Plot right side
    _plot_single_roc_set(axes[1], rocs2, colors2, legend_title2)
    # axes[1].tick_params(axis='y', labelleft=False) # Hide y-axis tick labels on the right plot (handled by sharey=True)

    # --- Final Touches ---
    # Add titles if desired (optional)
    # axes[0].set_title('ROC Curves - Set 1', fontsize=14)
    # axes[1].set_title('ROC Curves - Set 2', fontsize=14)

    plt.tight_layout() # Adjust layout to prevent overlap

def relabel_rocs(rocs, experiment_caption_prefix, experiment_n_vals):
    new_rocs = {}
    saved_keys = list(rocs.keys())
    for i, key in enumerate(saved_keys):
        n_val = experiment_n_vals[i]
        new_rocs[experiment_caption_prefix + str(n_val)] = rocs[key]
    return new_rocs

# calculate_roc_curves(experiment_dirs, experiment_caption_prefix, experiment_n_vals)


# original dataset size: 30+24=54
# training size: 0.9x n
# ×x
rocs_filename = 'classifiers/roc_curves_n100-1k.pkl'
colors = ['r', '#ff5500', '#ff7f00', '#ffaa00', 'y', 'g', '#00aaaa', 'c', '#0000ff', '#8000ff']
experiment_dirs = ['Exp_1000','Exp_1001','Exp_1002','Exp_1003','Exp_1004','Exp_1005','Exp_1006','Exp_1007','Exp_1008','Exp_1009']
experiment_n_vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
experiment_n_vals = [f'{i*0.9/54:.1f}×' for i in experiment_n_vals]
experiment_caption_prefix = ''
legend_title = 'Augmentation Factor (AUC)'

rocs = load_roc_curves(rocs_filename)
new_rocs = relabel_rocs(rocs, experiment_caption_prefix, experiment_n_vals)

# plot_roc_curves_with_bounds(new_rocs, limit_curves=3, legend_title=legend_title)

colors_first = colors
colors = ['r', '#ff7f00', 'y', 'g', 'c', '#0000ff', '#8000ff', '#8000ff']
rocs_filename = 'classifiers/roc_curves_samples2-30.pkl'
experiment_dirs = ['Exp_100','Exp_101','Exp_102','Exp_103','Exp_104','Exp_105','Exp_106','Exp_107']
experiment_n_vals = [2,3,5,10,15,20,25,30]
experiment_caption_prefix = ''
legend_title2 = 'Samples (AUC)'

rocs2 = load_roc_curves(rocs_filename)
new_rocs2 = relabel_rocs(rocs2, experiment_caption_prefix, experiment_n_vals)

# plot_roc_curves_with_bounds(new_rocs2, limit_curves=3, legend_title=legend_title2)

plot_rocs_sidebyside(new_rocs, new_rocs2, colors_first, colors, legend_title, legend_title2, limit_curves=3)

# plot_all_roc_curves(rocs)
# plot_roc_curves_with_bounds(new_rocs, limit_curves=3, legend_title=legend_title)
# plot_roc_curves_with_fpr_bounds(rocs)
plt.show()