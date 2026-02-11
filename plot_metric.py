import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Modern color palette (bright colors for dark background)
COLORS = ['#58a6ff', '#3fb950', '#a371f7', '#f0883e', '#f85149', '#39c5cf', '#db61a2', '#d29922']

def set_plot_style():
    """Sets a modern dark theme for matplotlib plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'figure.facecolor': '#0d1117',
        'figure.edgecolor': '#0d1117',
        'figure.dpi': 150,
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
        'grid.color': '#21262d',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'legend.fontsize': 11,
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
        'legend.framealpha': 0.9,
        'text.color': '#c9d1d9',
        'font.family': 'sans-serif',
        'font.size': 11,
        'savefig.facecolor': '#0d1117',
        'savefig.edgecolor': '#0d1117',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.3,
    })


def fetch_run_data(run_path):
    """Fetch history data from a wandb run."""
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    print(f"Fetching data for run: {run.name} ({run_path})")

    history = run.history(samples=10000, pandas=True)
    if history.empty:
        print("history() returned empty, trying scan_history()...")
        history = pd.DataFrame(list(run.scan_history()))

    print(f"Fetched {len(history)} data points")
    return run, history


def plot_metric(run_paths, metrics, output_dir="imgs", x_axis="step", smooth=0, title=None):
    """
    Plot one or more metrics from one or more wandb runs.

    Args:
        run_paths: List of run paths (entity/project/run_id)
        metrics: List of metric names to plot (e.g., "loss_balance/local_contribution_pct")
        output_dir: Directory to save plots
        x_axis: "step" or "epoch"
        smooth: Rolling window size for smoothing (0 = no smoothing)
        title: Custom plot title (auto-generated if None)
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    fig, ax = plt.subplots()
    color_idx = 0

    for run_path in run_paths:
        run, history = fetch_run_data(run_path)

        for metric in metrics:
            if metric not in history.columns:
                print(f"  WARNING: '{metric}' not found in run {run.name}")
                available = [c for c in history.columns if not c.startswith('_')]
                print(f"  Available metrics: {sorted(available)}")
                continue

            data = history[history[metric].notna()].copy()
            if data.empty:
                print(f"  WARNING: '{metric}' has no data in run {run.name}")
                continue

            # Determine x-axis
            if x_axis == "epoch" and 'epoch' in data.columns and data['epoch'].notna().any():
                data = data.sort_values('_step' if '_step' in data.columns else 'epoch')
                data = data.groupby('epoch')[metric].last().reset_index()
                x = data['epoch']
                xlabel = 'Epoch'
            else:
                x = data['_step'] if '_step' in data.columns else data.index
                xlabel = 'Step'

            y = data[metric]

            # Label: run_name / metric (simplified if only one run or one metric)
            if len(run_paths) == 1 and len(metrics) == 1:
                label = run.name
            elif len(run_paths) == 1:
                label = metric.split('/')[-1]
            elif len(metrics) == 1:
                label = run.name
            else:
                label = f"{run.name} / {metric.split('/')[-1]}"

            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1

            if smooth > 0 and len(y) > smooth:
                # Raw data as faint line
                ax.plot(x, y, color=color, linewidth=0.8, alpha=0.3)
                # Smoothed line
                y_smooth = y.rolling(window=smooth, min_periods=1).mean()
                ax.plot(x, y_smooth, color=color, linewidth=2.5, label=label)
            else:
                ax.plot(x, y, color=color, linewidth=2.5, label=label)

            print(f"  Plotted {metric} from {run.name}: {len(data)} points, range [{y.min():.4f}, {y.max():.4f}]")

    # Title
    if title:
        ax.set_title(title)
    elif len(metrics) == 1:
        ax.set_title(metrics[0])
    else:
        ax.set_title(' & '.join(m.split('/')[-1] for m in metrics))

    ax.set_xlabel(xlabel)
    ax.legend(loc='best')
    ax.set_ylabel(', '.join(m.split('/')[-1] for m in metrics))

    # Generate filename
    metric_slug = '_'.join(m.replace('/', '_') for m in metrics)
    run_slug = '_'.join(run_paths[0].split('/')[-1][:8] for _ in run_paths[:1])
    filename = f"{metric_slug}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()
    print(f"\nSaved: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot any metric from wandb runs with dark theme.",
        epilog="Examples:\n"
               "  python plot_metric.py user/project/run_id -m loss_balance/local_contribution_pct\n"
               "  python plot_metric.py run1 run2 -m val/loss --smooth 20\n"
               "  python plot_metric.py run1 -m train/step_loss val/loss --x epoch\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_paths", nargs='+', help="One or more wandb run paths (entity/project/run_id)")
    parser.add_argument("-m", "--metrics", nargs='+', required=True, help="Metric name(s) to plot")
    parser.add_argument("-o", "--output_dir", default="imgs", help="Output directory (default: imgs)")
    parser.add_argument("-x", "--x_axis", choices=["step", "epoch"], default="step", help="X-axis type (default: step)")
    parser.add_argument("--smooth", type=int, default=0, help="Rolling average window size (default: 0 = no smoothing)")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")

    args = parser.parse_args()
    plot_metric(args.run_paths, args.metrics, args.output_dir, args.x_axis, args.smooth, args.title)
