import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

pr_cache = {}

def compute_pagerank_from_W(W_path, alpha=0.85, max_iter=50):
    """Computes PageRank vector pi directly from saved sparse matrix *_W.pt."""
    W = torch.load(W_path, map_location="cpu")
    if W.layout == torch.sparse_csr:
        W = W.to_sparse_coo()
    W = W.coalesce()
    n = W.shape[0]
    row, col = W.indices()[0], W.indices()[1]
    val = W.values().double()
    
    pr = torch.full((n,), 1.0 / n, dtype=torch.double)
    for _ in range(max_iter):
        # Power iteration: pr_new = alpha * (W^T @ pr) + (1 - alpha) / n
        msg = val * pr[row]
        new_pr = torch.zeros(n, dtype=torch.double)
        new_pr.scatter_add_(0, col, msg)
        pr = alpha * new_pr + (1.0 - alpha) / n
    return pr.numpy()

def analyze_saved_pt(pt_filepath):
    """Computes Pearson correlations, tier shifts, and budget splits for a single run."""
    payload = torch.load(pt_filepath, map_location="cpu")
    fname = os.path.basename(pt_filepath)
    parts = fname.replace(".pt", "").split("_")
    dataset = parts[0]
    scenario = parts[1]
    seed = int(parts[2].replace("seed", ""))
    
    p_opt = payload['p_opt'].detach().cpu().numpy()
    q_opt = payload['q_opt'].detach().cpu().numpy()
    p0 = payload['p0'].detach().cpu().numpy()
    q0 = payload['q0'].detach().cpu().numpy()
    s = payload['s'].detach().cpu().numpy()
    
    # Locate <dataset>_W.pt to compute or load PageRank
    if dataset not in pr_cache:
        search_paths = [
            f"data/processed/{dataset}_W.pt",
            f"data/{dataset}_W.pt",
            f"results/{dataset}_W.pt",
            f"{dataset}_W.pt"
        ]
        for cand in search_paths:
            if os.path.exists(cand):
                pr_cache[dataset] = compute_pagerank_from_W(cand)
                break
        if dataset not in pr_cache:
            pr_cache[dataset] = None
            
    pr = pr_cache[dataset]
    
    dp = p_opt - p0
    dq = q_opt - q0
    budget_p = np.abs(dp)
    budget_q = np.abs(dq)
    total_node_budget = budget_p + budget_q
    
    extremity = np.abs(s - 0.5)
    r_dp_ext, _ = pearsonr(dp, extremity)
    r_dq_ext, _ = pearsonr(dq, extremity)
    
    if pr is not None and len(pr) == len(dp):
        r_bud_pr, _ = pearsonr(total_node_budget, pr)
    else:
        r_bud_pr = np.nan

    # Identify user tiers according to the 90-9-1 rule
    broad_mask = np.isclose(p0, 0.9, atol=1e-2)
    contr_mask = np.isclose(p0, 0.5, atol=1e-2)
    obser_mask = np.isclose(p0, 0.1, atol=1e-2)
    
    broad_dp, broad_dq = np.mean(dp[broad_mask]), np.mean(dq[broad_mask])
    contr_dp, contr_dq = np.mean(dp[contr_mask]), np.mean(dq[contr_mask])
    obser_dp, obser_dq = np.mean(dp[obser_mask]), np.mean(dq[obser_mask])
    
    total_b = np.sum(total_node_budget)
    pct_p = (np.sum(budget_p) / total_b) * 100.0 if total_b > 0 else 0.0
    pct_q = (np.sum(budget_q) / total_b) * 100.0 if total_b > 0 else 0.0
    
    return {
        'Dataset': dataset, 'Scenario': scenario, 'Seed': seed,
        'r(dp,e)': r_dp_ext, 'r(dq,e)': r_dq_ext, 'r(tau,pi)': r_bud_pr,
        'Broad_dp': broad_dp, 'Broad_dq': broad_dq,
        'Contr_dp': contr_dp, 'Contr_dq': contr_dq,
        'Obser_dp': obser_dp, 'Obser_dq': obser_dq,
        'pct_p': pct_p, 'pct_q': pct_q
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Table 4: RQ3 Node-Level Attribution and Correlations.")
    parser.add_argument("--budget", type=str, default="0.4", help="Budget ratio string to analyze (default: 0.4)")
    parser.add_argument("--dir", type=str, default="results/tensors", help="Directory where .pt tensor files are stored")
    parser.add_argument("--save_csv", type=str, default="results/rq3_table4_attribution.csv", help="Output path for summary CSV")
    args = parser.parse_args()

    patterns = [
        os.path.join(args.dir, f"*_B{args.budget}_CLEAR_PGD.pt"),
        f"results/*_B{args.budget}_CLEAR_PGD.pt",
        f"*_B{args.budget}_CLEAR_PGD.pt"
    ]
    
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(list(set(files)))
    
    if not files:
        print(f"No result files matching budget B={args.budget} found in '{args.dir}' or 'results/'.")
        print("Please ensure run_main.py has executed or check tensor file locations.")
        exit(0)

    print(f"Found {len(files)} result files. Computing correlations and tier statistics...")
    records = [analyze_saved_pt(f) for f in files]
    df = pd.DataFrame(records)
    
    dataset_order = ['wiki', 'gnutella', 'epinions', 'slashdot', 'youtube', 'pokec']
    df['order'] = df['Dataset'].apply(lambda x: dataset_order.index(x.lower()) if x.lower() in dataset_order else 99)
    df = df.sort_values(by=['order', 'Scenario']).drop(columns=['order'])
    
    summary = df.groupby(['Dataset', 'Scenario'], as_index=False).mean(numeric_only=True)
    summary = summary.drop(columns=['Seed'], errors='ignore')

    print("\n" + "="*110)
    print("TABLE 4: Attribution and Pearson Correlations at b/n = 0.4 (Averaged Across Seeds)")
    print("="*110)
    
    display_df = summary.copy()
    formatters = {
        'r(dp,e)': lambda x: f"{x:+.2f}",
        'r(dq,e)': lambda x: f"{x:+.2f}",
        'r(tau,pi)': lambda x: f"{x:+.2f}" if not np.isnan(x) else "N/A",
        'Broad_dp': lambda x: f"{x:+.2f}",
        'Broad_dq': lambda x: f"{x:+.2f}",
        'Contr_dp': lambda x: f"{x:+.2f}",
        'Contr_dq': lambda x: f"{x:+.2f}",
        'Obser_dp': lambda x: f"{x:+.2f}",
        'Obser_dq': lambda x: f"{x:+.2f}",
        'pct_p': lambda x: f"{x:.1f}%",
        'pct_q': lambda x: f"{x:.1f}%"
    }
    
    for col, fmt in formatters.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)
            
    print(display_df.to_string(index=False))
    print("="*110)
    
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    summary.to_csv(args.save_csv, index=False)
    print(f"\nSummary table saved to: {args.save_csv}")
