
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def set_plot_style():
    """Sets a modern dark theme for matplotlib plots."""
    # Use dark background style
    plt.style.use('dark_background')

    plt.rcParams.update({
        # Figure
        'figure.figsize': (12, 7),
        'figure.facecolor': '#0d1117',
        'figure.edgecolor': '#0d1117',
        'figure.dpi': 150,

        # Axes
        'axes.facecolor': '#0d1117',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        'axes.labelsize': 13,
        'axes.labelpad': 10,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.color': '#21262d',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,

        # Ticks
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',

        # Lines
        'lines.linewidth': 2.5,
        'lines.markersize': 8,

        # Legend
        'legend.fontsize': 11,
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
        'legend.framealpha': 0.9,

        # Text
        'text.color': '#c9d1d9',

        # Font
        'font.family': 'sans-serif',
        'font.size': 11,

        # Save
        'savefig.facecolor': '#0d1117',
        'savefig.edgecolor': '#0d1117',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.3,
    })


# Modern color palette (bright colors for dark background)
COLORS = {
    'blue': '#58a6ff',
    'green': '#3fb950',
    'purple': '#a371f7',
    'orange': '#f0883e',
    'red': '#f85149',
    'cyan': '#39c5cf',
    'pink': '#db61a2',
    'yellow': '#d29922',
}

def plot_metrics_from_run(run_path, output_dir="imgs"):
    """
    Downloads metrics from a specific wandb run and generates plots.

    Args:
        run_path (str): The path to the wandb run (e.g., "user/project/run_id").
        output_dir (str): Directory to save the plots.
    """
    print(f"Fetching data for run: {run_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        api = wandb.Api(timeout=60)
        run = api.run(run_path)

        # Try multiple methods to get history data
        print("Fetching history data...")

        # Method 1: Use history() with high sample count
        history = run.history(samples=10000, pandas=True)

        if history.empty:
            # Method 2: Try scan_history without key filtering
            print("history() returned empty, trying scan_history()...")
            all_rows = list(run.scan_history())
            history = pd.DataFrame(all_rows)

        print(f"Fetched {len(history)} data points")
        print(f"Columns: {history.columns.tolist()}")

        if history.empty:
            print("ERROR: No data found in this run.")
            print("The run might still be in progress or have no logged metrics.")
            return

    except Exception as e:
        print(f"Error fetching data from wandb: {e}")
        print("Please ensure the run path is correct and you are logged into wandb.")
        import traceback
        traceback.print_exc()
        return

    set_plot_style()

    # Debug: Show what validation data looks like
    print(f"\nAvailable columns containing 'val': {[c for c in history.columns if 'val' in c.lower()]}")
    print(f"Available columns containing 'loss': {[c for c in history.columns if 'loss' in c.lower()]}")

    if 'val/loss' in history.columns:
        val_debug = history[history['val/loss'].notna()][['_step', 'epoch', 'val/loss']].head(10) if 'epoch' in history.columns else history[history['val/loss'].notna()][['_step', 'val/loss']].head(10)
        print(f"\nValidation loss data sample:\n{val_debug}\n")

    # --- 1. Training & Validation Loss ---
    print("Generating Training & Validation Loss plot...")

    # Check if val/loss exists
    if 'val/loss' not in history.columns:
        print("Skipping Loss plot: 'val/loss' column not found in data.")
        print("This might be because the run is still in progress or uses different metric names.")
        val_loss_data = pd.DataFrame()  # Empty dataframe
    else:
        # For validation loss, we need rows where val/loss is not NaN
        # These are logged once per epoch at the end of validation
        val_loss_data = history[history['val/loss'].notna()].copy()

    # If epoch column is missing or all NaN in val rows, create epoch from row index
    if val_loss_data.empty or len(val_loss_data) == 0:
        print("Skipping Loss plot: No validation loss data found.")
    else:
        if 'epoch' not in val_loss_data.columns or val_loss_data['epoch'].isna().all():
            # Create epoch numbers from 0 to n-1
            val_loss_data['epoch'] = range(len(val_loss_data))
            print("Warning: 'epoch' column missing, using row index as epoch.")

        # Group by epoch and take the LAST value per epoch (handles duplicates)
        val_loss_data = val_loss_data.sort_values('_step' if '_step' in val_loss_data.columns else 'epoch')
        val_loss_data = val_loss_data.groupby('epoch')['val/loss'].last().reset_index()
        val_loss_data = val_loss_data.sort_values('epoch').reset_index(drop=True)

        print(f"  val/loss: {len(val_loss_data)} points")
        print(f"    First 3: {val_loss_data['val/loss'].head(3).tolist()}")
        print(f"    Last 3: {val_loss_data['val/loss'].tail(3).tolist()}")

        # Check if 'train/epoch_loss' was logged directly
        train_loss_data = None
        train_loss_label = None

        if 'train/epoch_loss' in history.columns and history['train/epoch_loss'].notna().any():
            train_loss_data = history[history['train/epoch_loss'].notna()].copy()
            if 'epoch' not in train_loss_data.columns or train_loss_data['epoch'].isna().all():
                train_loss_data['epoch'] = range(len(train_loss_data))
            # Group by epoch and take last value
            train_loss_data = train_loss_data.sort_values('_step' if '_step' in train_loss_data.columns else 'epoch')
            train_loss_data = train_loss_data.groupby('epoch')['train/epoch_loss'].last().reset_index()
            train_loss_data = train_loss_data.sort_values('epoch').reset_index(drop=True)
            train_loss_label = 'Training Loss (per Epoch)'
            train_loss_col = 'train/epoch_loss'

        elif 'train/step_loss' in history.columns and history['train/step_loss'].notna().any():
            # Calculate epoch-averaged training loss from step loss
            step_data = history[history['train/step_loss'].notna()].copy()
            if 'epoch' in step_data.columns and step_data['epoch'].notna().any():
                # Use LAST value per epoch (end-of-epoch loss) instead of mean
                step_data = step_data.sort_values('_step' if '_step' in step_data.columns else 'epoch')
                train_loss_data = step_data.groupby('epoch')['train/step_loss'].last().reset_index()
                train_loss_data.columns = ['epoch', 'train/epoch_loss_avg']
                train_loss_data = train_loss_data.sort_values('epoch').reset_index(drop=True)
                train_loss_label = 'Training Loss (End of Epoch)'
                train_loss_col = 'train/epoch_loss_avg'

        # Plot
        fig, ax = plt.subplots()
        if train_loss_data is not None and not train_loss_data.empty:
            ax.plot(train_loss_data['epoch'], train_loss_data[train_loss_col],
                    label=train_loss_label, color=COLORS['blue'], marker='o', markersize=5)
            ax.fill_between(train_loss_data['epoch'], train_loss_data[train_loss_col],
                            alpha=0.1, color=COLORS['blue'])
        ax.plot(val_loss_data['epoch'], val_loss_data['val/loss'],
                label='Validation Loss', color=COLORS['orange'], marker='s', markersize=5)
        ax.fill_between(val_loss_data['epoch'], val_loss_data['val/loss'],
                        alpha=0.1, color=COLORS['orange'])
        ax.set_title('Training & Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, f"{run.name}_train_val_loss.png"))
        plt.close()
        print(f"  Saved: {run.name}_train_val_loss.png")

    # Helper function for epoch-based plots
    def plot_epoch_metric(history_df, metric_col, title, ylabel, color_key, filename, marker='o'):
        if metric_col not in history_df.columns:
            print(f"Skipping {title}: '{metric_col}' not found.")
            return

        data = history_df[history_df[metric_col].notna()].copy()
        if data.empty:
            print(f"Skipping {title}: No data for '{metric_col}'.")
            return

        # Handle missing epoch column
        if 'epoch' not in data.columns or data['epoch'].isna().all():
            data['epoch'] = range(len(data))

        # Group by epoch and take the LAST value (end of epoch metrics)
        # This handles cases where multiple values might be logged per epoch
        data = data.sort_values('_step' if '_step' in data.columns else 'epoch')
        data = data.groupby('epoch')[metric_col].last().reset_index()
        data = data.sort_values('epoch').reset_index(drop=True)

        # Debug: show first and last few values
        print(f"  {metric_col}: {len(data)} points, range [{data[metric_col].min():.2f}, {data[metric_col].max():.2f}]")
        print(f"    First 3: {data[metric_col].head(3).tolist()}")
        print(f"    Last 3: {data[metric_col].tail(3).tolist()}")

        fig, ax = plt.subplots()
        ax.plot(data['epoch'], data[metric_col],
                label=ylabel, color=COLORS[color_key], marker=marker, markersize=5,
                linewidth=2.5)

        # Add subtle fill under the line
        ax.fill_between(data['epoch'], data[metric_col],
                        alpha=0.1, color=COLORS[color_key])

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        plt.savefig(os.path.join(output_dir, f"{run.name}_{filename}.png"))
        plt.close()
        print(f"  Saved: {run.name}_{filename}.png")

    # --- 2. Combined Metric ---
    print("Generating Combined Metric plot...")
    plot_epoch_metric(history, 'val/combined_metric',
                      'Validation Combined Metric (Recall + AUC)',
                      'Combined Metric', 'green', 'combined_metric', marker='s')

    # --- 3. Mean Retrieval Recall ---
    print("Generating Mean Recall plot...")
    plot_epoch_metric(history, 'val/mean_recall',
                      'Validation Mean Retrieval Recall (R@K)',
                      'Mean Recall (%)', 'purple', 'mean_recall', marker='^')

    # --- 4. Mean AUC ---
    print("Generating Mean AUC plot...")
    plot_epoch_metric(history, 'val/auc',
                      'Validation Mean Classification AUC',
                      'Mean AUC', 'cyan', 'mean_auc', marker='D')

    # --- 5. Learning Rate ---
    print("Generating Learning Rate plot...")
    # Learning rate might be logged as 'train/learning_rate' or 'learning_rate'
    lr_col = None
    if 'train/learning_rate' in history.columns and history['train/learning_rate'].notna().any():
        lr_col = 'train/learning_rate'
    elif 'learning_rate' in history.columns and history['learning_rate'].notna().any():
        lr_col = 'learning_rate'

    if lr_col and '_step' in history.columns:
        lr_df = history.dropna(subset=[lr_col, '_step']).copy()
        if not lr_df.empty:
            fig, ax = plt.subplots()
            ax.plot(lr_df['_step'], lr_df[lr_col],
                    label='Learning Rate', color=COLORS['red'], linewidth=2)
            ax.fill_between(lr_df['_step'], lr_df[lr_col],
                            alpha=0.15, color=COLORS['red'])
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Learning Rate')
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax.legend(loc='upper right')
            plt.savefig(os.path.join(output_dir, f"{run.name}_learning_rate.png"))
            plt.close()
            print(f"  Saved: {run.name}_learning_rate.png")
        else:
            print("Skipping Learning Rate plot: No valid data points.")
    else:
        print("Skipping Learning Rate plot: 'learning_rate' not found or empty.")
        
    # --- 6. Learned Temperature (τ = 1 / exp(logit_scale)) ---
    print("Generating Temperature plot...")
    if 'model/temperature' in history.columns and history['model/temperature'].notna().any():
        temp_data = history[history['model/temperature'].notna()].copy()
        if 'epoch' not in temp_data.columns or temp_data['epoch'].isna().all():
            temp_data['epoch'] = range(len(temp_data))
        temp_data = temp_data.sort_values('_step' if '_step' in temp_data.columns else 'epoch')
        temp_data = temp_data.groupby('epoch')['model/temperature'].last().reset_index()
        temp_data = temp_data.sort_values('epoch').reset_index(drop=True)
        # Convert exp(logit_scale) -> effective temperature τ = 1/exp(logit_scale)
        temp_data['temperature'] = 1.0 / temp_data['model/temperature']

        _, ax = plt.subplots()
        ax.plot(temp_data['epoch'], temp_data['temperature'],
                label='Effective Temperature (τ)', color=COLORS['yellow'], marker='o', markersize=5,
                linewidth=2.5)
        ax.fill_between(temp_data['epoch'], temp_data['temperature'],
                        alpha=0.1, color=COLORS['yellow'])
        ax.set_title('Learned Contrastive Temperature')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Temperature (τ)')
        # Format y-axis as plain decimals
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax.legend(loc='best')
        plt.savefig(os.path.join(output_dir, f"{run.name}_temperature.png"))
        plt.close()
        print(f"  τ range: [{temp_data['temperature'].min():.5f}, {temp_data['temperature'].max():.5f}]")
        print(f"  Saved: {run.name}_temperature.png")
    else:
        print("Skipping Temperature plot: 'model/temperature' not found.")

    # --- 7. Efficiency plots (step-based, not epoch-based) ---
    # Custom efficiency metrics logged by the trainer
    efficiency_cols = ['efficiency/step_time_ms', 'efficiency/avg_step_time_ms', 'efficiency/throughput_img_per_sec']
    has_efficiency = any(c in history.columns and history[c].notna().any() for c in efficiency_cols)

    if not has_efficiency:
        print("Custom efficiency columns not in history(), trying scan_history()...")
        try:
            eff_rows = list(run.scan_history(keys=efficiency_cols + ['_step'], page_size=10000))
            if eff_rows:
                eff_df = pd.DataFrame(eff_rows)
                has_efficiency = any(c in eff_df.columns and eff_df[c].notna().any() for c in efficiency_cols)
            else:
                eff_df = pd.DataFrame()
        except Exception as e:
            print(f"  scan_history failed: {e}")
            eff_df = pd.DataFrame()
    else:
        eff_df = history

    # Helper for step-based plots (used for both custom and system metrics)
    def plot_step_metric(df, col, title, ylabel, color_key, filename, x_col='_step', xlabel='Training Step'):
        if col not in df.columns or df[col].isna().all():
            print(f"  Skipping {title}: '{col}' not found or empty.")
            return
        data = df.dropna(subset=[col]).copy()
        if data.empty:
            return

        # Convert timestamps to elapsed minutes for readability
        if x_col == '_timestamp' and '_timestamp' in data.columns:
            t0 = data['_timestamp'].iloc[0]
            x = (data['_timestamp'] - t0) / 60.0  # elapsed minutes
            xlabel = 'Elapsed Time (min)'
        elif x_col in data.columns:
            x = data[x_col]
        else:
            x = data.index

        _, ax = plt.subplots()
        ax.plot(x, data[col], color=COLORS[color_key], linewidth=1.5, alpha=0.6)
        # Add smoothed line (rolling mean)
        if len(data) > 20:
            smoothed = data[col].rolling(window=20, min_periods=1).mean()
            ax.plot(x, smoothed, color=COLORS[color_key], linewidth=2.5, label=f'{ylabel} (smoothed)')
        else:
            ax.plot(x, data[col], color=COLORS[color_key], linewidth=2.5, label=ylabel)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        plt.savefig(os.path.join(output_dir, f"{run.name}_{filename}.png"))
        plt.close()
        print(f"  Saved: {run.name}_{filename}.png")

    if has_efficiency and not eff_df.empty:
        print("Generating custom efficiency plots...")
        plot_step_metric(eff_df, 'efficiency/throughput_img_per_sec',
                         'Training Throughput', 'Images / Second', 'orange', 'throughput')
        plot_step_metric(eff_df, 'efficiency/avg_step_time_ms',
                         'Average Step Time', 'Time (ms)', 'cyan', 'step_time')
    else:
        print("No custom efficiency metrics found.")

    # --- 7. W&B System Metrics (GPU memory, utilization, etc.) ---
    # These are automatically tracked by wandb under a separate stream.
    print("Fetching W&B system metrics...")
    try:
        sys_metrics = run.history(stream='events', samples=5000, pandas=True)
        if sys_metrics is not None and not sys_metrics.empty:
            sys_cols = [c for c in sys_metrics.columns if any(
                k in c.lower() for k in ['gpu', 'memory', 'cpu', 'disk', 'network', 'temp']
            )]
            if sys_cols:
                print(f"  Found system metrics: {sys_cols}")

            # GPU Memory
            gpu_mem_col = next((c for c in sys_metrics.columns if 'gpu' in c.lower() and 'memory' in c.lower()), None)
            if gpu_mem_col:
                print("Generating GPU Memory plot...")
                plot_step_metric(sys_metrics, gpu_mem_col,
                                 'GPU Memory Usage', 'Memory (%)', 'blue', 'gpu_memory',
                                 x_col='_timestamp', xlabel='Time')

            # GPU Utilization
            gpu_util_col = next((c for c in sys_metrics.columns
                                 if 'gpu' in c.lower() and ('util' in c.lower() or c.lower().endswith('gpu.0.gpu'))), None)
            if gpu_util_col:
                print("Generating GPU Utilization plot...")
                plot_step_metric(sys_metrics, gpu_util_col,
                                 'GPU Utilization', 'Utilization (%)', 'green', 'gpu_utilization',
                                 x_col='_timestamp', xlabel='Time')
        else:
            print("  No system metrics found for this run.")
    except Exception as e:
        print(f"  Could not fetch system metrics: {e}")

    print(f"\nPlots saved to '{output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from a Weights & Biases run.")
    parser.add_argument(
        "run_path", 
        type=str,
        help="Path to the wandb run in the format 'entity/project/run_id'."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="imgs",
        help="Directory to save the generated plots."
    )
    args = parser.parse_args()
    
    # Check for dependencies
    try:
        import importlib
        importlib.import_module('pandas')
        importlib.import_module('matplotlib')
    except ImportError:
        print("This script requires 'pandas' and 'matplotlib'.")
        print("Please install them using: pip install pandas matplotlib")
        exit(1)

    plot_metrics_from_run(args.run_path, args.output_dir)

