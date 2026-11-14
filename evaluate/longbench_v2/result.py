"""
Score LongBench V2 evaluation results.
Reads JSONL files from the results directory and reports accuracy
broken down by difficulty (easy/hard) and length (short/medium/long).

Usage:
    python evaluate/longbench_v2/result.py
    python evaluate/longbench_v2/result.py --results-dir evaluate/longbench_v2/results
"""
import os
import json
import argparse


def score_file(filepath):
    """Compute accuracy breakdown for a single JSONL result file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            pred_data = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None

    if not pred_data:
        return None

    # Deduplicate by _id, keeping the last occurrence
    seen = {}
    for item in pred_data:
        seen[item.get('_id', id(item))] = item
    pred_data = list(seen.values())

    counts = {
        'easy': [0, 0], 'hard': [0, 0],       # [total, correct]
        'short': [0, 0], 'medium': [0, 0], 'long': [0, 0]
    }

    for pred in pred_data:
        acc = int(pred.get('judge', False))

        # By difficulty
        diff = pred.get('difficulty', 'hard')
        if diff in counts:
            counts[diff][0] += 1
            counts[diff][1] += acc

        # By length
        length = pred.get('length', 'medium')
        if length in counts:
            counts[length][0] += 1
            counts[length][1] += acc

    total = len(pred_data)
    total_acc = sum(1 for p in pred_data if p.get('judge', False))

    def pct(n, d):
        return round(100 * n / d, 1) if d > 0 else 0.0

    return {
        'total': total,
        'overall': pct(total_acc, total),
        'easy': pct(counts['easy'][1], counts['easy'][0]),
        'hard': pct(counts['hard'][1], counts['hard'][0]),
        'short': pct(counts['short'][1], counts['short'][0]),
        'medium': pct(counts['medium'][1], counts['medium'][0]),
        'long': pct(counts['long'][1], counts['long'][0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Score LongBench V2 results")
    parser.add_argument("--results-dir", type=str, default="evaluate/longbench_v2/results")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    files = sorted([f for f in os.listdir(args.results_dir) if f.endswith('.jsonl')])
    if not files:
        print("No result files found.")
        return

    header = f"{'Model':<40s} {'Overall':>8s} {'Easy':>8s} {'Hard':>8s} {'Short':>8s} {'Medium':>8s} {'Long':>8s} {'N':>6s}"
    print(header)
    print('-' * len(header))

    output_lines = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\tN"]

    for filename in files:
        filepath = os.path.join(args.results_dir, filename)
        scores = score_file(filepath)
        if scores is None:
            continue

        name = filename.rsplit('.', 1)[0]
        row = f"{name:<40s} {scores['overall']:>7.1f}% {scores['easy']:>7.1f}% {scores['hard']:>7.1f}% {scores['short']:>7.1f}% {scores['medium']:>7.1f}% {scores['long']:>7.1f}% {scores['total']:>6d}"
        print(row)
        output_lines.append(f"{name}\t{scores['overall']}\t{scores['easy']}\t{scores['hard']}\t{scores['short']}\t{scores['medium']}\t{scores['long']}\t{scores['total']}")

    # Also save as TSV
    result_path = os.path.join(args.results_dir, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
