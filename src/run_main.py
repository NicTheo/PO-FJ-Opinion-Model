import os
import argparse
import time
import gc
import pandas as pd
import numpy as np
import torch

from graph_utils import setup_scenario
from clear_optimizer import clear_pgd
from baselines import (
    compute_pagerank,
    random_intervention,
    ideological_heuristic,
    bridge_heuristic,
)
from po_fj_dynamics import (
    compute_expected_opinions,
    get_K1_mask,
    compute_covariance_K1,
    compute_conflict,
)

def evaluate_state(W_sparse, s, a, p, q):
    """Evaluates the stationary network conflict, polarization, and disagreement."""
    mu, W_bar = compute_expected_opinions(W_sparse, s, a, p, q, T_mu=40)
    M_sparse = compute_covariance_K1(W_sparse, mu, p, q, a, get_K1_mask(W_sparse), T_M=15)
    C, P, D = compute_conflict(mu, M_sparse, W_bar)
    return C.item(), P.item(), D.item()

def save_vectors(graph_name, scenario, seed, ratio, method, s, p0, q0, p_opt, q_opt):
    """Saves node-level vectors for RQ3 attribution analysis."""
    os.makedirs("results/tensors", exist_ok=True)
    safe_method = method.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"results/tensors/{graph_name}_{scenario}_seed{seed}_B{ratio:.1f}_{safe_method}.pt"
    
    data = {
        's': s.detach().cpu(),
        'p0': p0.detach().cpu(),
        'q0': q0.detach().cpu(),
        'p_opt': p_opt.detach().cpu(),
        'q_opt': q_opt.detach().cpu()
    }
    torch.save(data, filename)

def run_experiment(graph_file, scenario="A", seeds=[42, 43, 44, 45, 46], heuristics_only=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_name = os.path.basename(graph_file).split('_')[0]
    
    print(f"\n{'='*55}\nRunning {graph_name.upper()} | Scenario {scenario} on {device}\n{'='*55}")
    
    W_sparse = torch.load(graph_file).to(device)
    if not W_sparse.is_sparse:
        W_sparse = W_sparse.to_sparse()
    W_sparse = W_sparse.coalesce()
    n = W_sparse.shape[0]
    
    # Precompute PageRank vector once per graph
    pi = compute_pagerank(W_sparse)
    
    budget_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    metrics_results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        s, p0, q0, a = setup_scenario(W_sparse, scenario_type=scenario)
        
        for ratio in budget_ratios:
            B = ratio * n
            print(f"  Budget: {ratio:.1f}n ({B:.0f})")
            
            # Baseline 0: Initial Unmodified State
            if ratio == 0.0:
                t0 = time.time()
                C, P, D = evaluate_state(W_sparse, s, a, p0, q0)
                eval_time = time.time() - t0
                metrics_results.append((graph_name, scenario, seed, ratio, 'Initial', C, P, D, eval_time, np.nan))
                save_vectors(graph_name, scenario, seed, ratio, "Initial", s, p0, q0, p0, q0)
                continue
            
            if not heuristics_only:
                # 1. CLEAR-PGD (Full joint optimization)
                pF, qF, _, t_tot, t_iter = clear_pgd(W_sparse, s, a, p0, q0, B, mode="full", T_opt=25)
                C, P, D = evaluate_state(W_sparse, s, a, pF, qF)
                metrics_results.append((graph_name, scenario, seed, ratio, 'CLEAR-PGD', C, P, D, t_tot, t_iter))
                save_vectors(graph_name, scenario, seed, ratio, "CLEAR-PGD", s, p0, q0, pF, qF)
                
                # 2. CLEAR-PGD (p-only)
                pp, qp, _, t_tot, t_iter = clear_pgd(W_sparse, s, a, p0, q0, B, mode="p-only", T_opt=25)
                C, P, D = evaluate_state(W_sparse, s, a, pp, qp)
                metrics_results.append((graph_name, scenario, seed, ratio, 'CLEAR-PGD (p-only)', C, P, D, t_tot, t_iter))
                save_vectors(graph_name, scenario, seed, ratio, "CLEAR-PGD (p-only)", s, p0, q0, pp, qp)
                
                # 3. CLEAR-PGD (q-only)
                pq, qq, _, t_tot, t_iter = clear_pgd(W_sparse, s, a, p0, q0, B, mode="q-only", T_opt=25)
                C, P, D = evaluate_state(W_sparse, s, a, pq, qq)
                metrics_results.append((graph_name, scenario, seed, ratio, 'CLEAR-PGD (q-only)', C, P, D, t_tot, t_iter))
                save_vectors(graph_name, scenario, seed, ratio, "CLEAR-PGD (q-only)", s, p0, q0, pq, qq)
                
                # 4. Random Intervention
                t0 = time.time()
                pr, qr = random_intervention(p0, q0, B)
                t_tot = time.time() - t0
                C, P, D = evaluate_state(W_sparse, s, a, pr, qr)
                metrics_results.append((graph_name, scenario, seed, ratio, 'Random', C, P, D, t_tot, np.nan))
                save_vectors(graph_name, scenario, seed, ratio, "Random", s, p0, q0, pr, qr)
            
            # 5. PageRank-Weighted Ideological-Heuristic
            t0 = time.time()
            ph_id, qh_id = ideological_heuristic(s, p0, q0, B, pi=pi)
            t_tot = time.time() - t0
            C, P, D = evaluate_state(W_sparse, s, a, ph_id, qh_id)
            metrics_results.append((graph_name, scenario, seed, ratio, 'Ideological-Heuristic', C, P, D, t_tot, np.nan))
            save_vectors(graph_name, scenario, seed, ratio, "Ideological-Heuristic", s, p0, q0, ph_id, qh_id)
            
            # 6. PageRank-Weighted Bridge-Heuristic
            t0 = time.time()
            ph_br, qh_br = bridge_heuristic(W_sparse, s, p0, q0, B, pi=pi)
            t_tot = time.time() - t0
            C, P, D = evaluate_state(W_sparse, s, a, ph_br, qh_br)
            metrics_results.append((graph_name, scenario, seed, ratio, 'Bridge-Heuristic', C, P, D, t_tot, np.nan))
            save_vectors(graph_name, scenario, seed, ratio, "Bridge-Heuristic", s, p0, q0, ph_br, qh_br)

    df = pd.DataFrame(metrics_results, columns=[
        'Dataset', 'Scenario', 'Seed', 'BudgetRatio', 'Method', 
        'Conflict', 'Polarization', 'Disagreement', 'TotalTime', 'TimePerIter'
    ])
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLEAR-PGD and baseline benchmarks.")
    parser.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "gnutella", "epinions", "slashdot", "youtube", "pokec", "all"], help="Dataset to evaluate")
    parser.add_argument("--scenario", type=str, default="both", choices=["A", "B", "both"], help="Opinion distribution scenario")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--heuristics_only", action="store_true", help="Only run heuristic baselines")
    parser.add_argument("--out_csv", type=str, default="results/main_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    all_graphs = ["wiki", "gnutella", "epinions", "slashdot", "youtube", "pokec"]
    target_graphs = all_graphs if args.dataset == "all" else [args.dataset]
    scenarios = ["A", "B"] if args.scenario == "both" else [args.scenario]
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tensors", exist_ok=True)

    for g_name in target_graphs:
        graph_path = os.path.join("data/processed", f"{g_name}_W.pt")
        if not os.path.exists(graph_path):
            print(f"Warning: {graph_path} not found. Skipping {g_name}...")
            continue
            
        for sc in scenarios:
            torch.cuda.empty_cache()
            gc.collect()
            try:
                df_res = run_experiment(graph_path, scenario=sc, seeds=args.seeds, heuristics_only=args.heuristics_only)
                
                # Append or create CSV
                if os.path.exists(args.out_csv):
                    df_res.to_csv(args.out_csv, mode='a', header=False, index=False)
                else:
                    df_res.to_csv(args.out_csv, index=False)
                    
                print(f"Saved results for {g_name} (Scenario {sc}) to {args.out_csv}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA Out of Memory on {g_name} (Scenario {sc}). Skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
