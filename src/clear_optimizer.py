import torch
import time
from torch.utils.checkpoint import checkpoint 
from po_fj_dynamics import compute_expected_opinions, compute_covariance_K1, compute_conflict, get_K1_mask

def clear_pgd(W_sparse, s, a, p0, q0, B, mode="full", T_opt=25, lr=0.1):
    """
    Projected Gradient Descent for Continuous-CLEAR.
    Uses PyTorch Gradient Checkpointing to prevent OOM on massive graphs.
    """
    device = W_sparse.device
    n = s.shape[0]
    
    p_opt = p0.clone().requires_grad_(True)
    q_opt = q0.clone().requires_grad_(True)
    
    # We only optimize the variables allowed by the mode
    opt_params = []
    if mode in ["full", "p-only"]: opt_params.append(p_opt)
    if mode in ["full", "q-only"]: opt_params.append(q_opt)
        
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    K1_mask = get_K1_mask(W_sparse)
    
    t0_total = time.time()
    
    # --- Define the Checkpoint Wrapper Function ---
    def objective_forward(p_in, q_in):
        mu, W_bar = compute_expected_opinions(W_sparse, s, a, p_in, q_in, T_mu=40)
        M_sparse = compute_covariance_K1(W_sparse, mu, p_in, q_in, a, K1_mask, T_M=15)
        C, _, _ = compute_conflict(mu, M_sparse, W_bar)
        return C

    for t in range(T_opt):
        optimizer.zero_grad()
        Conflict = checkpoint(objective_forward, p_opt, q_opt, use_reentrant=False)
        
        Conflict.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Projection step
        with torch.no_grad():
            if mode == "p-only":
                p_proj, _ = project_bounds_and_budget(p_opt, q0, p0, q0, B)
                p_opt.copy_(p_proj)
                q_opt.copy_(q0)
            elif mode == "q-only":
                _, q_proj = project_bounds_and_budget(p0, q_opt, p0, q0, B)
                p_opt.copy_(p0)
                q_opt.copy_(q_proj)
            else:
                p_proj, q_proj = project_bounds_and_budget(p_opt, q_opt, p0, q0, B)
                p_opt.copy_(p_proj)
                q_opt.copy_(q_proj)

    t_tot = time.time() - t0_total
    t_iter = t_tot / T_opt
    
    return p_opt.detach(), q_opt.detach(), Conflict.item(), t_tot, t_iter

def project_bounds_and_budget(p, q, p0, q0, B, max_proj_iters=3):
    """
    Step 4 of Algorithm 1: Enforces the [0,1] bounds and the L1 budget B.
    Uses an iterative scaling approach common in PGD adversarial attacks.
    """
    p_proj = p.clone()
    q_proj = q.clone()
    
    for _ in range(max_proj_iters):
        # 1. Enforce Box Bounds
        p_proj = torch.clamp(p_proj, 0.0, 1.0)
        q_proj = torch.clamp(q_proj, 0.0, 1.0)
        
        # 2. Enforce L1 Budget
        diff_p = p_proj - p0
        diff_q = q_proj - q0
        current_budget = torch.sum(torch.abs(diff_p)) + torch.sum(torch.abs(diff_q))
        
        if current_budget > B:
            # Scale differences back to exactly match budget b
            scale = B / current_budget
            p_proj = p0 + diff_p * scale
            q_proj = q0 + diff_q * scale
            
    return p_proj.detach(), q_proj.detach()

if __name__ == "__main__":
    import os
    from graph_utils import setup_scenario
    
    test_file = "data/wiki_W.pt" 
    
    if os.path.exists(test_file):
        print(f"=== Testing CLEAR-PGD Optimizer on {test_file} ===")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load graph
        W_sparse = torch.load(test_file).to(device)
        if not W_sparse.is_sparse:
            W_sparse = W_sparse.to_sparse()
            
        n = W_sparse.shape[0]
        
        # Setup Scenario A (Structurally Polarized)
        s, p0, q0, a = setup_scenario(W_sparse, scenario_type="A")
        
        # Set a 10% intervention budget (B = 0.1 * N)
        B = 0.1 * n 
        
        p_opt, q_opt, history = clear_pgd(W_sparse, s, a, p0, q0, B, mode="full", T_opt=20, lr=0.5)
        
        print("\n--- Optimization Results ---")
        print(f"Initial Conflict: {history[0]:.5f}")
        print(f"Final Conflict:   {history[-1]:.5f}")
        print(f"Conflict Reduction: {(history[0] - history[-1]) / history[0] * 100:.2f}%")
        
        # Verify L1 Budget was strictly respected
        total_changes = torch.sum(torch.abs(p_opt - p0)) + torch.sum(torch.abs(q_opt - q0))
        print(f"Total L1 Changes: {total_changes.item():.2f} (Max Allowed: {B:.2f})")
        print(f"Budget Strictly Respected? {total_changes.item() <= B + 1e-4}")
        
    else:
        print(f"Could not find {test_file}.")
