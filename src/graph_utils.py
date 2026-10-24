import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def compute_pagerank_pytorch(W_sparse, alpha=0.85, max_iter=100, tol=1e-6):
    """Computes PageRank using PyTorch sparse matrix multiplication."""
    n = W_sparse.shape[0]
    device = W_sparse.device
    dtype = W_sparse.dtype 
    
    # Initialize PageRank vector uniformly
    pr = torch.ones(n, 1, device=device, dtype=dtype) / n
    teleport = torch.ones(n, 1, device=device, dtype=dtype) / n
    
    for _ in range(max_iter):
        pr_next = alpha * torch.sparse.mm(W_sparse.t(), pr) + (1 - alpha) * teleport
        if torch.norm(pr_next - pr, p=1) < tol:
            break
        pr = pr_next
        
    return pr.squeeze()

def spectral_bisection(W_sparse):
    """Performs Spectral Bisection using Scipy's sparse eigenvalue solver."""
    # Convert PyTorch sparse to Scipy CSR for efficient eigenvalue decomposition
    indices = W_sparse._indices().cpu().numpy()
    values = W_sparse._values().cpu().numpy()
    shape = W_sparse.shape
    W_scipy = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    
    A = W_scipy + W_scipy.T
    
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    L_norm = sp.eye(shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    
    _, eigenvectors = eigsh(L_norm, k=2, which='SA')
    fiedler_vector = eigenvectors[:, 1]
    
    median_val = np.median(fiedler_vector)
    partition = (fiedler_vector >= median_val).astype(int)
    
    return torch.tensor(partition, device=W_sparse.device)

def setup_scenario(W_sparse, scenario_type="A"):
    """
    Sets up the initial s, p0, and q0 vectors for RQ2 using the 90-9-1 Rule.
    scenario_type: 'A' (Structurally Polarized) or 'B' (Emergent Random)
    """
    n = W_sparse.shape[0]
    device = W_sparse.device
    dtype = W_sparse.dtype  
    
    # 1. Inner Opinions (s)
    s = torch.zeros(n, device=device, dtype=dtype)
    if scenario_type == "A":
        partition = spectral_bisection(W_sparse)
        s[partition == 0] = torch.rand((partition == 0).sum(), device=device, dtype=dtype) * 0.3
        s[partition == 1] = 0.7 + torch.rand((partition == 1).sum(), device=device, dtype=dtype) * 0.3
    elif scenario_type == "B":
        s = torch.rand(n, device=device, dtype=dtype)
    else:
        raise ValueError("Scenario must be 'A' or 'B'")
        
    # 2. Vocal Influencers (p0, q0) using 90-9-1 Rule
    pr = compute_pagerank_pytorch(W_sparse)
    
    # Sort nodes by PageRank (descending)
    sorted_indices = torch.argsort(pr, descending=True)
    
    # Calculate indices for the tiers
    idx_1_percent = max(1, int(0.01 * n))
    idx_10_percent = max(idx_1_percent + 1, int(0.10 * n))
    
    creators = sorted_indices[:idx_1_percent]
    contributors = sorted_indices[idx_1_percent:idx_10_percent]
    observers = sorted_indices[idx_10_percent:]
    
    # Initialize default vectors
    p0 = torch.zeros(n, dtype=dtype, device=device)
    q0 = torch.zeros(n, dtype=dtype, device=device)
    
    # Top 1%: Broadcasters
    p0[creators] = 0.9
    q0[creators] = 0.1
    
    # Next 9%: Contributors
    p0[contributors] = 0.5
    q0[contributors] = 0.5
    
    # Bottom 90%: Observers
    p0[observers] = 0.1
    q0[observers] = 0.9
    
    # 3. Stubbornness (a)
    a = torch.full((n,), 0.5, device=device, dtype=dtype)
    
    return s, p0, q0, a

if __name__ == "__main__":
    import os
    import time
    
    test_file = "data/wiki_W.pt" 
    
    if os.path.exists(test_file):
        print(f"Loading {test_file}...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        W_sparse = torch.load(test_file).to(device)
        
        if not W_sparse.is_sparse:
            W_sparse = W_sparse.to_sparse()
            
        n = W_sparse.shape[0]
        print(f"Graph loaded on {device}. Nodes: {n}, Edges: {W_sparse._nnz()}")
        
        print("\n--- Testing Scenario A (Structurally Polarized) ---")
        t0 = time.time()
        sA, p0A, q0A, aA = setup_scenario(W_sparse, scenario_type="A")
        print(f"Setup time: {time.time() - t0:.2f}s")
        
        print("\n--- Verifying Node Assignments (90-9-1 Rule) ---")
        creators = (p0A == 0.9).sum().item()
        contributors = (p0A == 0.5).sum().item()
        observers = (p0A == 0.1).sum().item()
        
        print(f"Top 1% Creators (p=0.9): {creators} nodes")
        print(f"Next 9% Contributors (p=0.5): {contributors} nodes")
        print(f"Bottom 90% Observers (p=0.1): {observers} nodes")
        print(f"Total accounted for: {creators + contributors + observers} (Should be exactly {n})")
        print(f"Is stubbornness exactly 0.5 for ALL nodes? {torch.all(aA == 0.5).item()}")
    else:
        print(f"Could not find {test_file}.")
