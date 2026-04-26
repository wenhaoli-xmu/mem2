"""
Read heatmap.csv and draw individual heatmaps for each method.
Usage: python draw_heatmap.py [--csv heatmap.csv] [--output heatmap.png]
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.family'] = 'serif'

CMAP = LinearSegmentedColormap.from_list('custom', ['#d4edda', '#fff3cd', '#f8d7da', '#dc3545'])


def load_csv(csv_path):
    """Load heatmap.csv, return (batch_sizes, context_lengths, results_ours, results_baseline)."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    batch_sizes = sorted(set(int(r['batch_size']) for r in rows))
    context_lengths = sorted(set(int(r['context_length']) for r in rows))

    bs_idx = {bs: i for i, bs in enumerate(batch_sizes)}
    ctx_idx = {ctx: i for i, ctx in enumerate(context_lengths)}

    methods = sorted(set(r['method'] for r in rows))
    results = {}
    for m in methods:
        results[m] = np.full((len(batch_sizes), len(context_lengths)), np.nan)

    for r in rows:
        bi = bs_idx[int(r['batch_size'])]
        ci = ctx_idx[int(r['context_length'])]
        val = np.nan if r['ms_per_token'] == 'OOM' else float(r['ms_per_token'])
        results[r['method']][bi, ci] = val

    return batch_sizes, context_lengths, results


def format_ctx_label(ctx):
    if ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)


def draw_single_heatmap(data, title, batch_sizes, context_labels, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin, vmax = 0, 10

    im = ax.imshow(data, cmap=CMAP, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(context_labels)))
    ax.set_xticklabels(context_labels, fontsize=14)
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels(batch_sizes, fontsize=14)
    ax.set_xlabel('Context Length', fontsize=18)
    ax.set_ylabel('Batch Size', fontsize=18)
    ax.set_title(title, fontsize=18)

    for bi in range(len(batch_sizes)):
        for ci in range(len(context_labels)):
            val = data[bi, ci]
            if not np.isnan(val):
                color = 'white' if val > vmin + (vmax - vmin) * 0.85 else 'black'
                ax.text(ci, bi, f"{val:.2f}", ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold')
            else:
                ax.text(ci, bi, "OOM", ha='center', va='center',
                        color='red', fontsize=15, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ms/forward', fontsize=18)
    cbar.ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


METHOD_TITLES = {
    'ours': 'Ours (LRUCache)',
    'baseline': 'Baseline (FA2 + StaticCache)',
    'dynamic': 'Dynamic (FA2 + DynamicCache)',
}


def draw_heatmap(csv_path, output_path):
    batch_sizes, context_lengths, results = load_csv(csv_path)
    context_labels = [format_ctx_label(c) for c in context_lengths]
    base, ext = os.path.splitext(output_path)

    for method, data in results.items():
        if np.all(np.isnan(data)):
            continue
        # Convert ms/token → ms/forward by multiplying each row by its batch size
        fwd_data = data * np.array(batch_sizes)[:, None]
        title = METHOD_TITLES.get(method, method)
        draw_single_heatmap(fwd_data, title, batch_sizes, context_labels,
                            f"{base}_{method}{ext}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=os.path.join(script_dir, "heatmap.csv"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(script_dir, "heatmap.png"))
    args = parser.parse_args()

    draw_heatmap(args.csv, args.output)


if __name__ == '__main__':
    main()
