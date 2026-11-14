"""
Score LongBench V2 evaluation results.
Reads JSONL files from the results directory and reports accuracy
broken down by domain for each length category (short, medium, long).

Usage:
    python evaluate/longbench_v2/result_domain.py
    python evaluate/longbench_v2/result_domain.py --results-dir evaluate/longbench_v2/results
"""
import os
import json
import argparse


SHORT_NAMES = {
    'Single-Document QA': 'Single-QA',
    'Multi-Document QA': 'Multi-QA',
    'Code Repository Understanding': 'Code-Repo',
    'Long Structured Data Understanding': 'Struct-Data',
    'Long-dialogue History Understanding': 'Dialogue',
    'Long In-context Learning': 'ICL'
}


def short_name(d):
    return SHORT_NAMES.get(d, d[:11])


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

    domains = set()
    for pred in pred_data:
        domains.add(pred.get('domain', 'Unknown'))
        
    lengths = ['short', 'medium', 'long']
    counts = {l: {d: [0, 0] for d in domains} for l in lengths}

    for pred in pred_data:
        acc = int(pred.get('judge', False))
        domain = pred.get('domain', 'Unknown')
        length = pred.get('length', 'medium')
        if length in counts:
            counts[length][domain][0] += 1
            counts[length][domain][1] += acc

    def pct(n, d):
        return round(100 * n / d, 1) if d > 0 else 0.0

    scores_by_length = {}
    for l in lengths:
        total = sum(counts[l][d][0] for d in domains)
        total_acc = sum(counts[l][d][1] for d in domains)
        scores = {
            'total': total,
            'overall': pct(total_acc, total),
        }
        for d in domains:
            scores[d] = pct(counts[l][d][1], counts[l][d][0])
        scores_by_length[l] = scores

    return scores_by_length, domains


def main():
    parser = argparse.ArgumentParser(description="Score LongBench V2 results by domain")
    parser.add_argument("--results-dir", type=str, default="evaluate/longbench_v2/results")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    files = sorted([f for f in os.listdir(args.results_dir) if f.endswith('.jsonl')])
    if not files:
        print("No result files found.")
        return

    # First pass: collect all domains
    all_domains = set()
    file_scores = {}
    lengths = ['short', 'medium', 'long']
    
    for filename in files:
        filepath = os.path.join(args.results_dir, filename)
        res = score_file(filepath)
        if res is not None:
            scores_by_length, domains = res
            all_domains.update(domains)
            file_scores[filename] = scores_by_length

    sorted_domains = sorted(list(all_domains))
    
    domain_headers = " ".join([f"{short_name(d):>11s}" for d in sorted_domains])
    header = f"{'Model':<40s} {'Overall':>8s} {domain_headers} {'N':>6s}"
    
    output_lines = []

    for l in lengths:
        print(f"\n=== Length: {l.capitalize()} ===")
        print(header)
        print('-' * len(header))

        # For TSV file, use the full domain names
        output_lines.append(f"=== Length: {l.capitalize()} ===")
        output_lines.append("Model\tOverall\t" + "\t".join(sorted_domains) + "\tN")

        for filename in files:
            if filename not in file_scores:
                continue
                
            scores = file_scores[filename][l]
            name = filename.rsplit('.', 1)[0]
            
            row_domains = " ".join([f"{scores.get(d, 0.0):>10.1f}%" for d in sorted_domains])
            row = f"{name:<40s} {scores['overall']:>7.1f}% {row_domains} {scores['total']:>6d}"
            print(row)
            
            out_domains = "\t".join([str(scores.get(d, 0.0)) for d in sorted_domains])
            output_lines.append(f"{name}\t{scores['overall']}\t{out_domains}\t{scores['total']}")
            
        output_lines.append("")

    result_path = os.path.join(args.results_dir, 'result_domain.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
