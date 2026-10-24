import os
import argparse
import time
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from graph_utils import setup_scenario
from clear_optimizer import clear_pgd

def warmup_gpu_sparse(device):
    """Warms up CUDA kernels to ensure benchmark timing accuracy."""
    if device.type == 'cuda':
        print("Warming up GPU sparse kernels...")
        n = 5
        indices = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)
        values = torch.ones(5, device=device)
        W_sparse = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        
        s = torch.rand(n, device=device)
        a = torch.rand(n, device=device)
        p0 = torch.rand(n, device=device)
        q0 = torch.rand(n, device=device)
        
        _ = clear_pgd(W_sparse, s, a, p0, q0, B=1.0, mode="full", T_opt=2)
        torch.cuda.synchronize()
        print("GPU Warmup complete.\n")

def run_scalability_benchmark(device, seeds, budget_ratio, T_opt, out_csv):
    """Runs runtime scalability benchmarks across all available networks."""
    graph_names = ["gnutella", "wiki", "epinions", "slashdot", "youtube", "pokec"]
    results = []

    for name in graph_names:
        g_path = os.path.join("data/processed", f"{name}_W.pt")
        if not os.path.exists(g_path):
            g_path = os.path.join("data", f"{name}_W.pt")
        if not os.path.exists(g_path):
            print(f"Skipping {name}: file not found.")
            continue

        print(f"{'='*50}\nBenchmarking Scalability: {name.upper()}\n{'='*50}")
        try:
            t0 = time.time()
            W_sparse = torch.load(g_path, map_location=device)
            if not W_sparse.is_sparse:
                W_sparse = W_sparse.to_sparse()
            W_sparse = W_sparse.coalesce()
            
            n = W_sparse.shape[0]
            edges = W_sparse._nnz()
            print(f"Nodes: {n:,} | Edges: {edges:,} | Load Time: {time.time()-t0:.2f}s")

            seed_times = []
            for seed in seeds:
                torch.cuda.empty_cache()
                gc.collect()

                torch.manual_seed(seed)
                np.random.seed(seed)

                s, p0, q0, a = setup_scenario(W_sparse, scenario_type="B")
                B = budget_ratio * n

                _, _, _, _, t_iter = clear_pgd(W_sparse, s, a, p0, q0, B, mode="full", T_opt=T_opt)
                seed_times.append(t_iter)
                print(f"  Seed {seed} Time/Iter: {t_iter:.4f}s")

            mean_time = np.mean(seed_times)
            std_time = np.std(seed_times)
            total_time = mean_time * T_opt

            print(f"SUCCESS: {name} -> Mean Iter Time: {mean_time:.4f}s | Total ({T_opt} iter): {total_time:.4f}s\n")

            results.append({
                'Dataset': name,
                'Nodes': n,
                'Edges': edges,
                'TotalTime_Mean': total_time,
                'TimePerIter_Mean': mean_time,
                'TimePerIter_Std': std_time
            })
        except Exception as e:
            print(f"FAILED on {name}: {e}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Scalability records saved to: {out_csv}")
        return df
    return None

def plot_scalability(csv_file, T_opt=25):
    """Generates the Log-Log Figure 4 runtime scalability plot."""
    if not os.path.exists(csv_file):
        for alt in ["results/rq3_scalability_variance.csv", "rq3_scalability_variance.csv"]:
            if os.path.exists(alt):
                csv_file = alt
                break

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot generate plot: '{csv_file}' not found.")

    df = pd.read_csv(csv_file).sort_values(by="Edges")

    os.makedirs("results", exist_ok=True)
    os.makedirs("Plots", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    edges = df['Edges'].values
    times = df['TotalTime_Mean'].values
    errors = df['TimePerIter_Std'].values * T_opt
    names = df['Dataset'].values

    plt.errorbar(edges, times, yerr=errors,
                 marker='o', linestyle='-', color='crimson', markersize=8,
                 linewidth=2, capsize=4, elinewidth=1.5, label='CLEAR-PGD', zorder=5)

    for i, name in enumerate(names):
        name_lower = name.lower()
        if name_lower == 'gnutella':
            ha, offset = 'right', (-10, 8)
        elif name_lower == 'wiki':
            ha, offset = 'right', (-10, -14)
        elif name_lower == 'epinions':
            ha, offset = 'right', (-10, -12)
        elif name_lower == 'slashdot':
            ha, offset = 'left', (10, 4)
        elif name_lower == 'youtube':
            ha, offset = 'right', (-10, 10)
        else:  # pokec
            ha, offset = 'left', (10, -8)

        plt.annotate(f" {name.capitalize()}", (edges[i], times[i]),
                     textcoords="offset points", xytext=offset,
                     ha=ha, va='center', fontweight='bold', fontsize=10.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.3, 300)

    plt.xlabel(r'Number of Edges $|E|$', fontsize=12, fontweight='bold')
    plt.ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend(fontsize=11, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    out_pdf_results = "results/fig4_scalability.pdf"
    out_pdf_plots = "Plots/rq3_total_time_plot.pdf"
    
    plt.savefig(out_pdf_results, format='pdf', dpi=300)
    plt.savefig(out_pdf_plots, format='pdf', dpi=300)
    plt.close()
    print(f"SUCCESS: Figure saved to '{out_pdf_results}' and '{out_pdf_plots}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark and Plot RQ4 Scalability (Figure 4).")
    parser.add_argument("--plot_only", action="store_true", help="Only render the plot using existing CSV")
    parser.add_argument("--csv", type=str, default="results/rq4_scalability.csv", help="Scalability CSV path")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--budget_ratio", type=float, default=0.2, help="Budget ratio b/n")
    parser.add_argument("--T_opt", type=int, default=25, help="Optimization iterations")
    args = parser.parse_args()

    if not args.plot_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running Scalability Benchmarks on: {device}")
        warmup_gpu_sparse(device)
        run_scalability_benchmark(device, args.seeds, args.budget_ratio, args.T_opt, args.csv)

    plot_scalability(args.csv, T_opt=args.T_opt)
