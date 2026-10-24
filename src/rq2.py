import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10.5,
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

METHOD_STYLES = {
    'CLEAR-PGD': {'color': '#e41a1c', 'marker': '*', 'linestyle': '-', 'zorder': 10, 'label': 'CLEAR-PGD'},
    'CLEAR-PGD (p-only)': {'color': '#ff7f00', 'marker': 'X', 'linestyle': '-', 'zorder': 9},
    'CLEAR-PGD (q-only)': {'color': '#a65628', 'marker': '+', 'linestyle': '-', 'zorder': 8},
    'Bridge-Heuristic': {'color': '#984ea3', 'marker': 's', 'linestyle': '--', 'zorder': 5},
    'Ideological-Heuristic': {'color': '#377eb8', 'marker': '^', 'linestyle': '--', 'zorder': 4},
    'Random': {'color': '#999999', 'marker': 'o', 'linestyle': '--', 'zorder': 3}
}

def load_and_prep_data(filepath):
    """Loads benchmark CSV and aligns initial (b/n = 0) states across all methods."""
    df = pd.read_csv(filepath)
    metrics = ['Conflict', 'Polarization', 'Disagreement']
    
    initial_states = df[df['BudgetRatio'] == 0.0].copy()
    other_states = df[df['BudgetRatio'] > 0.0].copy()
    
    methods = [m for m in df['Method'].unique() if m != 'Initial']
    
    expanded_initials = []
    for method in methods:
        temp = initial_states.copy()
        temp['Method'] = method
        expanded_initials.append(temp)
        
    df_clean = pd.concat([other_states] + expanded_initials, ignore_index=True)
    
    agg_df = df_clean.groupby(['Dataset', 'Scenario', 'BudgetRatio', 'Method'])[metrics].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    return agg_df

def plot_single_dataset_gridspec(agg_df, target_dataset, filename):
    """Generates the 2x3 GridSpec plot for Scenario A and Scenario B."""
    metrics = [
        ('Conflict', r"Conflict ($\overline{\mathcal{C}}$)"),
        ('Polarization', r"Polarization ($\overline{\mathcal{P}}$)"),
        ('Disagreement', r"Disagreement ($\overline{\mathcal{D}}$)")
    ]
    scenarios = ['A', 'B']
    
    data = agg_df[agg_df['Dataset'].str.lower() == target_dataset.lower()]
    if data.empty:
        available = agg_df['Dataset'].unique().tolist()
        raise ValueError(f"Dataset '{target_dataset}' not found in data! Available datasets: {available}")
    
    fig = plt.figure(figsize=(15, 4.6))
    gs = GridSpec(3, 3, height_ratios=[0.12, 1, 1], hspace=0.32, wspace=0.28, 
                  left=0.07, right=0.98, top=0.96, bottom=0.12)
    
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r+1, c])
            
    handles_dict = {}
    
    for row, scenario in enumerate(scenarios):
        for col, (metric_col, title) in enumerate(metrics):
            ax = axes[row, col]
            scen_data = data[data['Scenario'] == scenario]
            
            for method in METHOD_STYLES.keys():
                method_data = scen_data[scen_data['Method'] == method].sort_values('BudgetRatio')
                if method_data.empty: 
                    continue
                
                style = METHOD_STYLES[method]
                x = method_data['BudgetRatio']
                y_mean = method_data[f'{metric_col}_mean']
                y_std = method_data[f'{metric_col}_std']
                lbl = style.get('label', method)
                
                line, = ax.plot(x, y_mean, color=style['color'], marker=style['marker'], 
                                linestyle=style['linestyle'], label=lbl, 
                                zorder=style['zorder'])
                
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=style['color'], 
                                alpha=0.15, zorder=style['zorder']-1)
                
                if method not in handles_dict:
                    handles_dict[method] = line
            
            if row == 0:
                ax.set_title(f"{title}", fontweight='bold', fontsize=12, pad=4)
                
            ax.set_ylabel(f"{title}", fontsize=10, fontweight='normal')
                
            if row == 1:
                ax.set_xlabel(r"Budget Ratio ($b/n$)", fontsize=10)
                
            ax.set_xlim([0, 1.0])
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    handles = [handles_dict[m] for m in METHOD_STYLES.keys() if m in handles_dict]
    labels = [METHOD_STYLES[m].get('label', m) for m in METHOD_STYLES.keys() if m in handles_dict]
    
    ax_legend.legend(handles, labels, loc='center', ncol=6, frameon=True, 
                     shadow=False, fontsize=10.5, edgecolor='black', 
                     handletextpad=0.4, columnspacing=0.9)

    pos_row0 = axes[0, 0].get_position()
    pos_row1 = axes[1, 0].get_position()
    y_row0 = pos_row0.y0 + pos_row0.height / 2
    y_row1 = pos_row1.y0 + pos_row1.height / 2

    fig.text(0.015, y_row0, "Scenario A", va='center', ha='center', rotation=90, fontweight='bold', fontsize=11)
    fig.text(0.015, y_row1, "Scenario B", va='center', ha='center', rotation=90, fontweight='bold', fontsize=11)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", filename)
    plt.savefig(out_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"SUCCESS: Saved RQ2 conflict figure for '{target_dataset}' to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 3: Conflict, Polarization, and Disagreement curves.")
    parser.add_argument("--dataset", type=str, default="epinions", help="Dataset to plot (default: epinions)")
    parser.add_argument("--csv", type=str, default="results/main_metrics.csv", help="Path to input metrics CSV")
    args = parser.parse_args()

    csv_candidates = [args.csv, "results/main_metrics_pagerank_heuristics.csv", "main_metrics_pagerank_heuristics.csv"]
    chosen_csv = None
    for c in csv_candidates:
        if os.path.exists(c):
            chosen_csv = c
            break
            
    if chosen_csv is None:
        raise FileNotFoundError(f"Could not find any metrics CSV. Checked: {csv_candidates}")

    out_pdf = f"fig3_{args.dataset.lower()}_conflict_mitigation.pdf"
    print(f"Loading data from {chosen_csv} to plot '{args.dataset}'...")
    agg_df = load_and_prep_data(chosen_csv)
    plot_single_dataset_gridspec(agg_df, target_dataset=args.dataset, filename=out_pdf)
