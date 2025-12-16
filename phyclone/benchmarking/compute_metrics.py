from tree_metrics import compute_metrics
import pandas as pd


def main(args):

    row = {
        "prog": args.program,
        "depth": args.depth,
        "num_mutations": args.num_mutations,
        "num_samples": args.num_samples,
    }

    metrics_df = compute_metrics(
        args.pred_tree_file,
        args.true_tree_file,
        stats_file=args.stats_file,
        row_init=row,
    )
    print("metrics computed ", metrics_df)
    metrics_df.to_csv(args.out_file, index=False, sep="\t")
    print("saved to ", args.out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pred-tree-file", required=True)

    parser.add_argument("-t", "--true-tree-file", required=True)

    parser.add_argument("-s", "--stats-file", default=None)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-P", "--program", required=True)

    parser.add_argument("-D", "--depth", type=int, required=True)

    parser.add_argument("-M", "--num-mutations", type=int, required=True)

    parser.add_argument("-S", "--num-samples", type=int, required=True)

    cli_args = parser.parse_args()

    main(cli_args)
