import torch
from clear_optimizer import project_bounds_and_budget

def compute_pagerank(W_sparse, alpha=0.85, max_iter=50, tol=1e-6):
    """
    Computes PageRank vector pi on the GPU using power iteration.
    W_sparse is row-stochastic, so we use W_sparse.T @ pi.
    """
    n = W_sparse.shape[0]
    device = W_sparse.device
    pi = torch.full((n,), 1.0 / n, dtype=W_sparse.dtype, device=device)
    teleport = (1.0 - alpha) / n
    
    W_T = W_sparse.transpose(0, 1).coalesce()
    
    for _ in range(max_iter):
        pi_next = alpha * torch.sparse.mm(W_T, pi.unsqueeze(1)).squeeze(1) + teleport
        if torch.norm(pi_next - pi, p=1) < tol:
            pi = pi_next
            break
        pi = pi_next
        
    return pi / pi.sum()

def random_intervention(p0, q0, B):
    """
    Applies uniform random perturbations to p and q, 
    exhausting the exact budget B.
    """
    dp = torch.rand_like(p0)
    dq = torch.rand_like(q0)
    
    total_rand = torch.sum(dp) + torch.sum(dq)
    if total_rand > 0:
        dp = dp * (B / total_rand)
        dq = dq * (B / total_rand)
    
    return project_bounds_and_budget(p0 + dp, q0 + dq, p0, q0, B)

def ideological_heuristic(s, p0, q0, B, pi=None):
    """
    PageRank-Weighted Ideological-Heuristic:
    - Extremists prioritized by pi_i * |s_i - 0.5| -> Observers (p=0, q=1)
    - Moderates prioritized by pi_i * (0.5 - |s_i - 0.5|) -> Broadcasters (p=1, q=0)
    Exhausts the budget B by alternating iteratively.
    """
    n = s.shape[0]
    extremity = torch.abs(s - 0.5)
    moderateness = 0.5 - extremity
    
    if pi is not None:
        ex_score = pi * extremity
        mod_score = pi * moderateness
    else:
        ex_score = extremity
        mod_score = moderateness
    
    p_opt = p0.clone()
    q_opt = q0.clone()
    
    extremist_indices = torch.argsort(ex_score, descending=True)
    moderate_indices = torch.argsort(mod_score, descending=True)
    
    budget_remaining = float(B)
    ex_ptr = 0
    mod_ptr = 0
    visited = set()
    
    while budget_remaining > 1e-7 and (ex_ptr < n or mod_ptr < n):
        # 1. Target one high-PageRank Extremist (p->0, q->1)
        while ex_ptr < n and extremist_indices[ex_ptr].item() in visited:
            ex_ptr += 1
            
        if ex_ptr < n:
            idx_ex = extremist_indices[ex_ptr].item()
            visited.add(idx_ex)
            cost_p = p_opt[idx_ex].item()
            cost_q = 1.0 - q_opt[idx_ex].item()
            
            if cost_p <= budget_remaining:
                p_opt[idx_ex] = 0.0
                budget_remaining -= cost_p
            else:
                p_opt[idx_ex] -= budget_remaining
                budget_remaining = 0.0
                
            if budget_remaining > 1e-7:
                if cost_q <= budget_remaining:
                    q_opt[idx_ex] = 1.0
                    budget_remaining -= cost_q
                else:
                    q_opt[idx_ex] += budget_remaining
                    budget_remaining = 0.0
            ex_ptr += 1
        
        if budget_remaining <= 1e-7:
            break
            
        # 2. Target one high-PageRank Moderate (p->1, q->0)
        while mod_ptr < n and moderate_indices[mod_ptr].item() in visited:
            mod_ptr += 1
            
        if mod_ptr < n:
            idx_mod = moderate_indices[mod_ptr].item()
            visited.add(idx_mod)
            cost_p = 1.0 - p_opt[idx_mod].item()
            cost_q = q_opt[idx_mod].item()
            
            if cost_p <= budget_remaining:
                p_opt[idx_mod] = 1.0
                budget_remaining -= cost_p
            else:
                p_opt[idx_mod] += budget_remaining
                budget_remaining = 0.0
                
            if budget_remaining > 1e-7:
                if cost_q <= budget_remaining:
                    q_opt[idx_mod] = 0.0
                    budget_remaining -= cost_q
                else:
                    q_opt[idx_mod] -= budget_remaining
                    budget_remaining = 0.0
            mod_ptr += 1

    return p_opt, q_opt

def bridge_heuristic(W_sparse, s, p0, q0, B, pi=None):
    """
    PageRank-Weighted Bridge-Heuristic:
    Identifies connected nodes with opposing views weighted by PageRank:
    score(i, j) = (pi_i + pi_j) * |s_i - s_j|.
    Forces them to listen (q -> 1) while leaving posting rates (p) untouched.
    """
    W_coal = W_sparse.coalesce()
    indices = W_coal.indices()
    row, col = indices[0], indices[1]
    
    # 1. Calculate ideological distance across every edge, weighted by PageRank
    edge_diff = torch.abs(s[row] - s[col])
    if pi is not None:
        edge_score = (pi[row] + pi[col]) * edge_diff
    else:
        edge_score = edge_diff
    
    # 2. Sort edges by score (descending)
    sorted_edge_idx = torch.argsort(edge_score, descending=True)
    
    p_opt = p0.clone() # p remains completely untouched
    q_opt = q0.clone()
    
    budget_remaining = float(B)
    processed_nodes = set()
    
    # 3. Iterate through the top-scoring edges and maximize q for those nodes
    for e_idx in sorted_edge_idx:
        if budget_remaining <= 1e-7:
            break
            
        u = row[e_idx].item()
        v = col[e_idx].item()
        
        # Process the higher-PageRank endpoint first
        if pi is not None and pi[v].item() > pi[u].item():
            u, v = v, u
        
        for node in (u, v):
            if budget_remaining <= 1e-7:
                break
            if node not in processed_nodes:
                cost_q = 1.0 - q_opt[node].item()
                if budget_remaining >= cost_q:
                    q_opt[node] = 1.0
                    budget_remaining -= cost_q
                else:
                    q_opt[node] += budget_remaining
                    budget_remaining = 0.0
                processed_nodes.add(node)
            
    return p_opt, q_opt

