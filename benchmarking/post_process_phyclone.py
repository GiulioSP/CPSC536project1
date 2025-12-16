import numpy as np
import pandas as pd
import networkx as nx
import skbio
import pickle


def write_pickle(graph, out_file):
    with open(out_file, "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def load_tree_from_results(phyclone_results_file, clonal_newick_file, remove_root=True, include_outlier_node=False):
    df = pd.read_table(phyclone_results_file)
    tree = skbio.read(clonal_newick_file, format="newick", into=skbio.TreeNode, convert_underscores=False)

    phyclone_tree = _load_tree_from_results_files_read(df, tree, remove_root, include_outlier_node=include_outlier_node)

    return phyclone_tree


def _load_tree_from_results_files_read(df, tree, remove_root, include_outlier_node):
    snvs = df.drop_duplicates(["mutation_id", "clone_id"]).groupby("clone_id")["mutation_id"].apply(set).to_dict()

    nx_tree = build_nx_tree_from_skbio_tree(tree)

    for n in nx_tree.nodes:
        if n in snvs:
            nx_tree.nodes[n]["snvs"] = snvs[n]
        else:
            nx_tree.nodes[n]["snvs"] = {}

    if include_outlier_node:
        if -1 in snvs:
            nx_tree.add_node(-1)
            nx_tree.nodes[-1]["snvs"] = snvs[-1]

            nx_tree.add_edge("root", -1)

    if remove_root:
        nx_tree.remove_node("root")
    else:
        nx_tree.graph["root_name"] = "root"

    return nx_tree


def build_nx_tree_from_skbio_tree(skbio_tree):
    nx_tree = nx.DiGraph()

    for node in skbio_tree.traverse():

        if node.name == "root":
            node_name = "root"
        else:
            node_name = int(node.name)

        if node.is_tip():
            nx_tree.add_node(node_name)

        for child in node.children:
            if child.name == "root":
                child_name = "root"
            else:
                child_name = int(child.name)
            nx_tree.add_edge(node_name, child_name)

    return nx_tree


def main(args):
    tree = load_tree_from_results(
        args.phyclone_table,
        args.clone_newick,
        remove_root=args.remove_root,
        include_outlier_node=args.include_outlier_node,
    )
    write_pickle(tree, args.out_tree)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--phyclone-table", required=True)

    parser.add_argument("-t", "--clone-newick", required=True)

    parser.add_argument("-o", "--out-tree", required=True)

    parser.add_argument("--remove-root", default=True)

    parser.add_argument("--include-outlier-node", default=False)

    cli_args = parser.parse_args()

    main(cli_args)
