import os
import torch

def generate_sbm_graph(n=2000, p_in=0.06, p_out=0.005, device='cpu'):
    """Generates a synthetic Stochastic Block Model (SBM) graph with realistic variance."""
    print(f"Generating synthetic SBM graph (n={n})...")
    half = n // 2
    membership = torch.zeros(n, dtype=torch.long, device=device)
    membership[half:] = 1
    
    rand_mat = torch.rand((n, n), device=device)
    in_block = (membership.unsqueeze(0) == membership.unsqueeze(1))
    p_mat = torch.where(in_block, p_in, p_out)
    adj = (rand_mat < p_mat).float()
    torch.diagonal(adj).zero_()
    
    row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
    W = adj / row_sums
    W_csr = W.to_sparse_csr()
    
    s = torch.where(membership == 0, torch.tensor(0.2, device=device), torch.tensor(0.8, device=device))
    s += torch.randn(n, device=device) * 0.1
    s = torch.clamp(s, 0.0, 1.0)
    
    a = torch.full((n,), 0.4, device=device, dtype=torch.float32)
    
    p0 = torch.full((n,), 0.5, device=device, dtype=torch.float32)
    q0 = torch.full((n,), 0.5, device=device, dtype=torch.float32)
    
    return W_csr, s, a, p0, q0

def compute_exact_dense_M(W_sparse, p, q, s, a, max_iters=20, mask=None):
    """
    Computes stationary covariance using Lemma 2 operators.
    If mask is None, computes exact dense M.
    If mask is provided, applies K-hop structural truncation at each iteration.
    """
    W = W_sparse.to_dense()
    n = W.shape[0]
    
    W_bar = torch.diag(q) @ (W @ torch.diag(p) + torch.diag(W @ (1 - p))) + torch.diag(1 - q)
    mu = s.clone()
    for _ in range(30):
        mu = torch.diag(a) @ s + torch.diag(1 - a) @ W_bar @ mu
        
    Sigma_x = torch.diag(p * (1 - p))
    Sigma_y = torch.diag(q * (1 - q))
    qqT = torch.ger(q, q)
    ppT = torch.ger(p, p)
    
    B_mu = W @ torch.diag(mu) - torch.diag(mu) @ W
    
    term1 = (Sigma_x + qqT) * (B_mu @ Sigma_y @ B_mu.t())
    term2 = Sigma_x * (B_mu @ ppT @ B_mu.t())
    I_A = torch.diag(1 - a)
    Omega = I_A @ (term1 + term2) @ I_A
    
    def G_op(M, C):
        WC = W_bar @ C
        CWt = C @ W_bar.t()
        WCWt = W_bar @ C @ W_bar.t()
        return W_bar @ (M * C) @ W_bar.t() - W_bar @ (M * CWt) - (WC * M) @ W_bar.t() + M * WCWt
        
    def D_op(M):
        return (Sigma_x + qqT) * G_op(M, Sigma_y) + Sigma_x * G_op(M, ppT)
        
    M = torch.zeros((n, n), device=W.device, dtype=W.dtype)
    for _ in range(max_iters):
        M_next = I_A @ (W_bar @ M @ W_bar.t() + D_op(M)) @ I_A + Omega
        if mask is not None:
            M_next = M_next * mask
        diff = torch.norm(M_next - M, p='fro')
        M = M_next
        if diff < 1e-6:
            break
            
    return M, W_bar, mu

def compute_conflict_metrics(W_bar, mu, M):
    """
    Computes Stationary Expected Polarization (Eq. 19), Disagreement (Eq. 20),
    and overall Conflict C = P + D (Eq. 21).
    """
    n = mu.shape[0]
    z_bar = mu.mean()
    diag_M = torch.diag(M)
    
    P_classic = ((mu - z_bar) ** 2).mean()
    P_cov = diag_M.mean() - M.sum() / (n ** 2)
    P_bar = P_classic + P_cov
    
    diff_mu_sq = (mu.unsqueeze(1) - mu.unsqueeze(0)) ** 2

    W_off = W_bar.clone()
    torch.diagonal(W_off).zero_()
    d_in = W_off.sum(dim=0)
    
    D_classic = 0.5 * (W_off * diff_mu_sq).sum() / n
    D_cov = 0.5 * ((1.0 + d_in) * diag_M).sum() / n - (W_off * M).sum() / n
    D_bar = D_classic + D_cov
    
    C_bar = P_bar + D_bar
    return P_bar.item(), D_bar.item(), C_bar.item()

def evaluate_graph(name, W_csr, s, a, p0, q0):
    device = W_csr.device
    n = W_csr.shape[0]
    print(f"\n{'='*60}")
    print(f"Evaluating {name} (n={n}) on {device}")
    print(f"{'='*60}")
    
    M_dense, W_bar, mu = compute_exact_dense_M(W_csr, p0, q0, s, a, mask=None)
    energy_dense = torch.norm(M_dense, p='fro')**2
    P_dense, D_dense, C_dense = compute_conflict_metrics(W_bar, mu, M_dense)
    
    print(f"Exact Dense Energy ||M||_F^2 = {energy_dense.item():.6f}")
    print(f"Exact Dense Metrics: P = {P_dense:.6f} | D = {D_dense:.6f} | Conflict C = {C_dense:.6f}")
    
    print("Building structural Hop masks (K=1, 2, 3)...")
    I_sparse = torch.sparse_coo_tensor(
        torch.stack([torch.arange(n, device=device), torch.arange(n, device=device)]), 
        torch.ones(n, device=device), size=(n, n)
    )
    
    # K=1 mask
    W_coo = W_csr.to_sparse_coo()
    M1_sparse = (W_coo + I_sparse).coalesce()
    M1_mask = (M1_sparse.to_dense() > 0).float()
    
    # K=2 mask
    M2_sparse = torch.sparse.mm(M1_sparse, M1_sparse).coalesce()
    M2_mask = (M2_sparse.to_dense() > 0).float()
    
    # K=3 mask
    M3_sparse = torch.sparse.mm(M2_sparse, M1_sparse).coalesce()
    M3_mask = (M3_sparse.to_dense() > 0).float()
    
    masks = {1: M1_mask, 2: M2_mask, 3: M3_mask}
    
    results = {}
    print(f"\n--- RESULTS FOR TABLE 2 ({name}) [Direct Masking] ---")
    for K in [1, 2, 3]:
        mask = masks[K]
        edges_k = mask.sum().item()
        sparsity = (edges_k / (n * n)) * 100
        
        M_k = M_dense * mask
        energy_k = torch.norm(M_k, p='fro')**2
        rho_k = (energy_k / energy_dense).item()
        
        P_k, D_k, C_k = compute_conflict_metrics(W_bar, mu, M_k)
        eps_C_pct = (abs(C_k - C_dense) / C_dense) * 100
        eps_D_pct = (abs(D_k - D_dense) / D_dense) * 100
        
        results[K] = (sparsity, rho_k, C_k, eps_C_pct)
        print(
            f"K={K}: Sparsity = {sparsity:6.2f}% | rho_K = {rho_k:.4f} | "
            f"C^(K) = {C_k:.6f} | Rel. Error eps_C = {eps_C_pct:.4f}% (eps_D = {eps_D_pct:.4f}%)"
        )
    print(
        f"Dense (Exact): Sparsity = 100.00% | rho_K = 1.0000 | "
        f"C^(dense) = {C_dense:.6f} | Rel. Error eps_C = 0.0000%"
    )

    print(f"\n--- RESULTS FOR TABLE 2 ({name}) [Iterative K-Hop Truncation] ---")
    for K in [1, 2, 3]:
        mask = masks[K]
        edges_k = mask.sum().item()
        sparsity = (edges_k / (n * n)) * 100
        
        M_k_iter, _, _ = compute_exact_dense_M(W_csr, p0, q0, s, a, mask=mask)
        energy_k_iter = torch.norm(M_k_iter, p='fro')**2
        rho_k_iter = (energy_k_iter / energy_dense).item()
        
        P_k_iter, D_k_iter, C_k_iter = compute_conflict_metrics(W_bar, mu, M_k_iter)
        eps_C_iter_pct = (abs(C_k_iter - C_dense) / C_dense) * 100
        eps_D_iter_pct = (abs(D_k_iter - D_dense) / D_dense) * 100
        
        print(
            f"K={K}: Sparsity = {sparsity:6.2f}% | rho_K = {rho_k_iter:.4f} | "
            f"C^(K) = {C_k_iter:.6f} | Rel. Error eps_C = {eps_C_iter_pct:.4f}% (eps_D = {eps_D_iter_pct:.4f}%)"
        )
    print()
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Wiki
    wiki_path = os.path.join("/project/home/p201599/continuous_clear/data", "wiki_W.pt")
    if os.path.exists(wiki_path):
        W_wiki = torch.load(wiki_path).to(device=device, dtype=torch.float32)
        state_wiki = torch.load(
            os.path.join("/project/home/p201599/continuous_clear/data/initial_states", "state_wiki_W.pt"),
            map_location=device
        )
        evaluate_graph(
            "Wiki",
            W_wiki,
            state_wiki["s"].to(torch.float32),
            state_wiki["a"].to(torch.float32),
            state_wiki["p0"].to(torch.float32),
            state_wiki["q0"].to(torch.float32)
        )

    # 2. Synthetic SBM
    W_sbm, s_sbm, a_sbm, p_sbm, q_sbm = generate_sbm_graph(n=2000, p_in=0.06, p_out=0.005, device=device)
    evaluate_graph("Synthetic SBM", W_sbm, s_sbm, a_sbm, p_sbm, q_sbm)
