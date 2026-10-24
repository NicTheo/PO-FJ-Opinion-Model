import torch

def get_expected_W(W_sparse, p, q):
    """
    Computes the expected interaction matrix \bar{W} (Definition 1).
    \bar{W} = diag(q)[W * diag(p) + diag(W(1-p))] + (I - diag(q))
    """
    n = W_sparse.shape[0]
    device = W_sparse.device
    
    # W_sparse is a constant input, so internal access is safe here
    indices = W_sparse._indices()
    values = W_sparse._values()
    row, col = indices[0], indices[1]
    
    # 1. Non-diagonal edge transfer: \bar{W}_{ij} = q_i * p_j * W_{ij}
    new_values = values * q[row] * p[col]
    W_qp = torch.sparse_coo_tensor(indices, new_values, (n, n)).coalesce()
    
    # 2. Diagonal accumulation: \bar{W}_{ii} = q_i * [sum_{k!=i}(1-p_k)W_{ik}] + (1 - q_i)
    one_minus_p = 1.0 - p
    edge_weights_1_p = values * one_minus_p[col]
    
    # Accumulate weights memory-safely
    W_1_p = torch.zeros(n, dtype=values.dtype, device=device)
    W_1_p.scatter_add_(0, row, edge_weights_1_p)
    
    diag_add = q * W_1_p + (1.0 - q)
    
    diag_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    W_diag = torch.sparse_coo_tensor(diag_indices, diag_add, (n, n)).coalesce()
    
    W_bar = (W_qp + W_diag).coalesce()
    return W_bar

def compute_expected_opinions(W_sparse, s, a, p, q, T_mu=50):
    """
    Implements Lemma 1: Iterative evaluation of \mu^{(t+1)}
    """
    W_bar = get_expected_W(W_sparse, p, q)
    mu = s.clone()
    
    W_bar_idx = W_bar.indices()
    W_bar_val = W_bar.values()
    row, col = W_bar_idx[0], W_bar_idx[1]
    
    for _ in range(T_mu):
        weighted_opinions = W_bar_val * mu[col]
        
        W_bar_mu = torch.zeros_like(mu)
        W_bar_mu.scatter_add_(0, row, weighted_opinions)
        
        mu = a * s + (1.0 - a) * W_bar_mu
        
    return mu, W_bar

def get_B_mu(W_sparse, mu):
    n = W_sparse.shape[0]
    indices = W_sparse._indices()
    values = W_sparse._values()
    row, col = indices[0], indices[1]
    B_values = values * (mu[col] - mu[row])
    return torch.sparse_coo_tensor(indices, B_values, (n, n)).coalesce()

def get_K1_mask(W_sparse):
    n = W_sparse.shape[0]
    indices = W_sparse._indices()
    mask_values = torch.ones_like(W_sparse._values())
    
    diag_indices = torch.arange(n, device=W_sparse.device).unsqueeze(0).repeat(2, 1)
    diag_values = torch.ones(n, device=W_sparse.device)
    
    all_indices = torch.cat([indices, diag_indices], dim=1)
    all_values = torch.cat([mask_values, diag_values], dim=0)
    
    mask = torch.sparse_coo_tensor(all_indices, all_values, (n, n)).coalesce()
    # Safely extract values
    mask_values = torch.clamp(mask.values(), max=1.0)
    return torch.sparse_coo_tensor(mask.indices(), mask_values, (n, n)).coalesce()

def compute_covariance_K1(W_sparse, mu, p, q, a, K1_mask, T_M=20):
    n = W_sparse.shape[0]
    device = W_sparse.device
    
    sigma_x = q * (1.0 - q)
    sigma_y = p * (1.0 - p)
    
    M_indices = K1_mask._indices()
    M_values = torch.zeros(M_indices.shape[1], device=device, dtype=W_sparse.dtype)
    row, col = M_indices[0], M_indices[1]
    
    for _ in range(T_M):
        decay = (1.0 - a[row]) * (1.0 - a[col])
        injection = sigma_x[row] * sigma_y[col] * torch.abs(mu[row] - mu[col])
        M_values = decay * M_values + injection

    M_sparse = torch.sparse_coo_tensor(M_indices, M_values, (n, n)).coalesce()
    return M_sparse

def compute_conflict(mu, M_sparse, W_bar_sparse):
    n = mu.shape[0]
    
    zeta = mu.mean()
    P_mu = torch.mean((mu - zeta)**2)
    
    W_bar_idx = W_bar_sparse.indices()
    W_bar_val = W_bar_sparse.values()
    row, col = W_bar_idx[0], W_bar_idx[1]
    
    diff_sq = (mu[row] - mu[col])**2
    D_mu = torch.sum(W_bar_val * diff_sq) / (2 * n)
    
    # Preserve M gradient
    M_idx = M_sparse.indices()
    M_val = M_sparse.values()
    
    diag_mask = M_idx[0] == M_idx[1]
    M_diag_sum = torch.sum(M_val[diag_mask])
    M_all_sum = torch.sum(M_val)
    
    P_M = (M_diag_sum / n) - (M_all_sum / (n**2))
    D_M = M_diag_sum / (2 * n) 
    
    Polarization = P_mu + P_M
    Disagreement = D_mu + D_M
    Conflict = Polarization + Disagreement
    
    return Conflict, Polarization, Disagreement
