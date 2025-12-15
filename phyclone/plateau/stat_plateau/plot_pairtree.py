import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent.parent.parent

# Get pairtree samples
# change paht based on window size of interest
pairtree_run_dir = project_root / "phyclone" / "plateau" / "stat_plateau" /"output" / "pairtree_run" / "window-20"
pairtree_samples = sorted([d.name for d in pairtree_run_dir.iterdir() if d.is_dir()])

# change end folder path as well depending on where you wanna save to
plot_dir = project_root / "phyclone" / "plateau" / "stat_plateau" / "output" / "pairtree_plots" / "window-twenty-python"
plot_dir.mkdir(parents=True, exist_ok=True)

def ESS_pairtree(dataset):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for i in range(1, 4):  # replicates 1, 2, 3
        dir_path = pairtree_run_dir / dataset / f"{dataset}_rep{i}"
        
        # Load data from subdirectories
        all_data = []
        for subdir in dir_path.iterdir():
            if subdir.is_dir():
                csv_files = list(subdir.glob("*.csv"))
                for csv_file in csv_files:
                    dat = pd.read_csv(csv_file)
                    
                    # Extract chain name from folder name (e.g., "k2_chains" -> "2")
                    name = subdir.name.split("_")[0].replace("k", "")
                    
                    dat_subset = dat[['window', 'between_chain_ess']].copy()
                    dat_subset['chains'] = name
                    all_data.append(dat_subset)
        
        if all_data:
            temp = pd.concat(all_data, ignore_index=True)
            
            # Plot on subplot
            ax = axes[i-1]
            colors = plt.cm.tab10(range(len(temp['chains'].unique())))
            for idx, chain in enumerate(sorted(temp['chains'].unique())):
                subset = temp[temp['chains'] == chain]
                ax.plot(subset['window'], subset['between_chain_ess'], label=f"Chains {chain}", color=colors[idx], linewidth=2)
            
            ax.set_xlabel("Window (size = 20)", fontsize=12)
            ax.set_ylabel("Between Chain ESS", fontsize=12)
            ax.set_title(f"Replicate {i}", fontsize=13)
            ax.grid(True, alpha=0.3)
            
            if i == 2:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=11)
    
    fig.subplots_adjust(bottom=0.25, top=0.88)
    fig.suptitle(f"Pairtree ESS plots for {dataset}", fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    return fig

# Create plots for each sample
for sample in pairtree_samples:
    fig = ESS_pairtree(sample)
    filename = f"{sample}_window20.png"
    fig.savefig(plot_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("All plots created successfully!")