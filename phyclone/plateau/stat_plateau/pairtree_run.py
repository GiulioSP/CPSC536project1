from stat_plateau import stat_plateau
import pandas as pd
import os
from pathlib import Path

# for analyzing pairtree data convergence

project_root = Path(__file__).parent.parent.parent.parent
wanted_lst = [
    "k10s30_T50_M100",
    # "k10s30_T50_M200",
    # "k10s30_T200_M200",
    # "k10s30_T200_M100",
    # "k10s30_T1000_M200",
    # "k10s100_T50_M100",
    # "k10s100_T50_M200",
    # "k10s100_T200_M100"
]
for item in wanted_lst:
    output_dir = project_root / "phyclone" / "plateau" / "stat_plateau" / "output" / "pairtree_run" / "full_chain" / item
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1,4):
        wanted = f"{item}_rep{i}"
        print(f"Running pairtree_run for {wanted}...\n")
        csv_path = project_root / "data" / "pairtree_subsets" / wanted / f"{wanted}.csv"

        # plateau checker for pairtree test 
        analyzer = stat_plateau(
            csv_path=str(csv_path),
            k=10,
            check_all_subsets=True,
            window=0,
            convergence_threshold=None
        )

        output_dir_plus = output_dir / wanted
        os.makedirs(output_dir_plus, exist_ok=True)

        analyzer.eval_convergence(output_base_dir=str(output_dir_plus))

        print(f"Completed pairtree_run for {wanted}.\n")
