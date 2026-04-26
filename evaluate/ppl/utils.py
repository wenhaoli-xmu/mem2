import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Lipari palette colors
LIPARI_COLORS = ['#1A2744', '#3B4F7A', '#7B6B85', '#B07B6B', '#D9A085', '#F0D0B0']

def ema_smooth(values, alpha=0.9999):
    """Compute exponential moving average and exponential moving std."""
    n = len(values)
    ema = np.zeros(n)
    ema_var = np.zeros(n)
    ema[0] = values[0]
    ema_var[0] = 0.0
    for i in range(1, n):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * values[i]
        diff = values[i] - ema[i]
        ema_var[i] = alpha * ema_var[i - 1] + (1 - alpha) * diff * diff
    sigma = np.sqrt(ema_var)
    return ema, sigma

def plot_ppl_curve(neg_logprobs, task_name, save_dir, filename_base, max_position_embeddings=262144):
    """Plot EMA-smoothed per-token NLL curve with ±σ band."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    color = LIPARI_COLORS[3]  # warm coral

    values = np.array(neg_logprobs)
    ema, sigma = ema_smooth(values, alpha=0.9995)
    x_axis = np.arange(len(ema))

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Rasterized scatter points so the PDF doesn't become huge
    ax.scatter(x_axis, values, s=1, color=color, alpha=0.01, rasterized=True)
    ax.plot(x_axis, ema, linewidth=2.5, color=color)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 15)
    ax.set_xlim(0, max_position_embeddings)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{filename_base}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  Saved plot to {save_dir}/{filename_base}.{{pdf,png}}', flush=True)

def plot_multiple_ppl_curves(json_paths, labels, colors, linestyles, save_dir, filename_base, max_position_embeddings=262144):
    """Plot EMA-smoothed per-token NLL curve for multiple methods."""
    import json
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    fig, ax = plt.subplots(figsize=(4, 3))
    
    for path, label, color, linestyle in zip(json_paths, labels, colors, linestyles):
        with open(path, 'r') as f:
            data = json.load(f)
        neg_logprobs = data["neg_logprobs"]
        values = np.array(neg_logprobs)
        ema, sigma = ema_smooth(values, alpha=0.9999)
        x_axis = np.arange(len(ema))
        
        # ax.scatter(x_axis, values, s=1, color=color, alpha=0.02, rasterized=True)
        ax.plot(x_axis, ema, linewidth=2.5, color=color, label=label, linestyle=linestyle)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 15)
    ax.set_xlim(0, max_position_embeddings)
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{filename_base}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  Saved combined plot to {save_dir}/{filename_base}.{{pdf,png}}', flush=True)

if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from spotlight.misc import get_env_conf

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--task", type=str, default="evaluate/ppl/perplexity_tasks.json")
    parser.add_argument("--max-position-embeddings", type=int, default=262144)
    args = parser.parse_args()

    test_conf = get_env_conf(args.task)
    vis_dir = f"evaluate/ppl/visualize/{args.model_name}"
    
    methods = [
        ("origin", LIPARI_COLORS[0], "-", "dense attention"), 
        ("hash-eval", LIPARI_COLORS[1], "-", "ours (Eager)"), 
        ("kernel", "#6079A3", "--", "ours (CUDA)")
    ]
    
    for task in test_conf:
        task_name = task["task_name"]
        prefix = task_name.split('.')[0]
        
        json_paths = []
        labels = []
        colors = []
        linestyles = []
        for method, color, linestyle, label in methods:
            p = os.path.join(vis_dir, f"{prefix}-{method}.json")
            if os.path.exists(p):
                json_paths.append(p)
                labels.append(label)
                colors.append(color)
                linestyles.append(linestyle)
        
        if json_paths:
            plot_multiple_ppl_curves(json_paths, labels, colors, linestyles, vis_dir, f"{prefix}-combined", args.max_position_embeddings)
