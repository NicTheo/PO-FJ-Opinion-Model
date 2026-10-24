import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sbm(n_nodes=100, p_in=0.15, p_out=0.02):
    sizes = [n_nodes // 2, n_nodes // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, directed=True, seed=42)
    
    while not nx.is_strongly_connected(G):
        G = nx.stochastic_block_model(sizes, probs, directed=True)
        
    A = nx.adjacency_matrix(G).toarray().astype(float)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    W = A / row_sums[:, np.newaxis]
    return G, W

def expected_PO_FJ_metrics(W, p, q, a_val, s):
    n = W.shape[0]
    A = np.eye(n) * a_val
    
    W_p = W @ np.diag(p)
    
    unposted_weights_per_node = np.sum(W @ np.diag(1.0 - p), axis=1)
    diag_W_1_p = np.diag(unposted_weights_per_node)
    
    W_bar = np.diag(q) @ (W_p + diag_W_1_p) + np.diag(1.0 - q)
    
    I = np.eye(n)
    inv_term = np.linalg.inv(I - (I - A) @ W_bar)
    mu_bar = inv_term @ A @ s
    
    public_opinion = np.mean(mu_bar)
    polarization = np.mean((mu_bar - public_opinion)**2)
    
    disagreement = 0.0
    for i in range(n):
        for j in range(n):
            if W_bar[i, j] > 0:
                disagreement += W_bar[i, j] * (mu_bar[i] - mu_bar[j])**2
    disagreement = disagreement / (2 * n)
    
    return public_opinion, polarization, disagreement

# 1. Setup Network
n = 100
G, W = generate_sbm(n)
s = np.zeros(n)
s[n//2:] = 1.0 # Blue=0, Red=1
a_val = 0.5

# --- Compute Classic FJ Baseline (p=1, q=1) ---
p_classic = np.ones(n)
q_classic = np.ones(n)
fj_po, fj_pol, fj_dis = expected_PO_FJ_metrics(W, p_classic, q_classic, a_val, s)
print(f"=== Classic FJ Baselines ===")
print(f"Public Opinion (Z_FJ): {fj_po:.4f}")
print(f"Polarization (P_FJ):   {fj_pol:.4f}")
print(f"Disagreement (D_FJ):   {fj_dis:.4f}\n")

# 2. Setup Grid
grid_size = 21 
p_vals = np.linspace(0.0, 1.0, grid_size)
q_vals = np.linspace(0.0, 1.0, grid_size)

PO_matrix = np.zeros((grid_size, grid_size))
Pol_matrix = np.zeros((grid_size, grid_size))
Dis_matrix = np.zeros((grid_size, grid_size))

# Fix Blue community behavior
p_blue = 0.5
q_blue = 0.5

# 3. Sweep Red community behavior
for i, q_r in enumerate(q_vals):
    for j, p_r in enumerate(p_vals):
        p = np.zeros(n)
        q = np.zeros(n)
        
        p[:n//2] = p_blue
        q[:n//2] = q_blue
        p[n//2:] = p_r
        q[n//2:] = q_r
        
        po, pol, dis = expected_PO_FJ_metrics(W, p, q, a_val, s)
        PO_matrix[i, j] = po
        Pol_matrix[i, j] = pol
        Dis_matrix[i, j] = dis

# 4. Create 1x4 Subplots
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# --- PLOT 0: Network Graph ---
np.random.seed(42)
initial_pos = {}
for node in G.nodes():
    if G.nodes[node]['block'] == 0: # Blue
        initial_pos[node] = [-2.0 + np.random.uniform(-0.5, 0.5), np.random.uniform(-1.0, 1.0)]
    else: # Red
        initial_pos[node] = [2.0 + np.random.uniform(-0.5, 0.5), np.random.uniform(-1.0, 1.0)]
        
pos = nx.spring_layout(G, pos=initial_pos, iterations=3, seed=42)

blue_nodes = [node for node in G.nodes() if G.nodes[node]['block'] == 0]
red_nodes = [node for node in G.nodes() if G.nodes[node]['block'] == 1]

nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, ax=axes[0], node_color='#1f77b4', node_size=60, alpha=0.9, edgecolors='white')
nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, ax=axes[0], node_color='#d62728', node_size=60, alpha=0.9, edgecolors='white')

G_undirected = G.to_undirected()
nx.draw_networkx_edges(G_undirected, pos, ax=axes[0], alpha=0.3, edge_color='gray')

axes[0].set_title("Stochastic Block Model Graph", fontsize=16)
axes[0].axis('off')

axes[0].text(0.25, -0.05, 
             r"$p=0.5, \; q=0.5$",
             ha='center', va='top', transform=axes[0].transAxes, 
             fontsize=14, color='#1f77b4', fontweight='bold',
             bbox=dict(fc="white", ec="none", alpha=0.9))
             
axes[0].text(0.75, -0.05, 
             r"$p_{\text{red}}, \; q_{\text{red}} \in [0,1]$",
             ha='center', va='top', transform=axes[0].transAxes, 
             fontsize=14, color='#d62728', fontweight='bold',
             bbox=dict(fc="white", ec="none", alpha=0.9))

# --- PLOT 1: Public Opinion ---
sns.heatmap(PO_matrix, ax=axes[1], cmap="coolwarm", center=0.5, cbar_kws={'label': ''})
axes[1].set_title(r"Public Opinion ($\overline{\mathcal{Z}}$)", fontsize=16)

# --- PLOT 2: Polarization (Flipped Colormap: dark for high) ---
sns.heatmap(Pol_matrix, ax=axes[2], cmap="magma_r", cbar_kws={'label': ''})
axes[2].set_title(r"Polarization ($\overline{\mathcal{P}}$)", fontsize=16)

# --- PLOT 3: Disagreement (Flipped Colormap: dark for high) ---
sns.heatmap(Dis_matrix, ax=axes[3], cmap="plasma_r", cbar_kws={'label': ''})
axes[3].set_title(r"Disagreement ($\overline{\mathcal{D}}$)", fontsize=16)

# Format Heatmap Axes
tick_locations = np.array([0, 4, 8, 12, 16, 20]) 
tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

for ax in axes[1:]:
    ax.invert_yaxis()
    ax.set_xticks(tick_locations + 0.5)
    ax.set_yticks(tick_locations + 0.5)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=12)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=12)
    ax.set_xlabel(r"$p_{\text{red}}$", fontsize=14)
    ax.set_ylabel(r"$q_{\text{red}}$", fontsize=14)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
out_file = os.path.join("results", "heatmap_sbm_final.pdf")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Saved fully compiled 1x4 figure to {out_file}")
