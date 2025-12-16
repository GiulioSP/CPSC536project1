from sklearn.metrics import homogeneity_completeness_v_measure
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import pickle


def compute_metrics(pred_tree_file, true_tree_file, stats_file=None, row_init=None):
    if row_init:
        row = row_init
    else:
        row = {}

    row, pred_tree = _compute_metrics_rows(row, pred_tree_file, true_tree_file)

    _add_stats_cols(row, stats_file)

    df = pd.DataFrame([row])

    return df


def _compute_metrics_rows(row, pred_tree_file, true_tree_file):
    
    pred_tree, true_tree = prep_trees(pred_tree_file, true_tree_file)
    
    row.update(compute_clustering_metrics_from_trees(pred_tree, true_tree))
    row.update(compute_ancestor_descendant_metrics(pred_tree, true_tree))
    row.update(compute_num_nodes(pred_tree, true_tree))

    return row, pred_tree


def _add_stats_cols(row, stats_file):
    if stats_file is not None:
        stats = pd.read_csv(stats_file, sep="\t")
        row.update(stats.iloc[0].to_dict())


def prep_trees(pred_tree_file, true_tree_file):
    pred_tree = read_pickle(pred_tree_file)
    true_tree = read_pickle(true_tree_file)
    pred_tree, true_tree = tree_snv_equalizer(pred_tree, true_tree)
    return pred_tree, true_tree


def compute_num_nodes(pred_tree, true_tree):
    result = {}

    num_nodes_pred = _get_num_nodes(pred_tree)
    num_nodes_true = _get_num_nodes(true_tree)

    result["num_nodes_pred"] = num_nodes_pred
    result["num_nodes_true"] = num_nodes_true
    return result


def _get_num_nodes(tree):
    num_nodes = 0
    for n in tree.nodes:
        if len(tree.nodes[n]["snvs"]) > 0:
            num_nodes += 1
    return num_nodes


def _get_union_snv_set_from_tree(tree):
    total_set = set()
    for node in tree.nodes:
        curr_snvs = set(tree.nodes[node]["snvs"])
        total_set.update(curr_snvs)
    return total_set


def tree_snv_equalizer(pred_tree, true_tree, add_loss_node=False):
    pred_snv_set = _get_union_snv_set_from_tree(pred_tree)

    true_snv_set = _get_union_snv_set_from_tree(true_tree)

    set_intersect = pred_snv_set.intersection(true_snv_set)

    _update_tree_with_filtered_snv_set(set_intersect, pred_tree)

    if add_loss_node:
        snvs_lost_by_pred = true_snv_set.difference(pred_snv_set)

        if not pred_tree.has_node(-1):
            pred_tree.add_node(-1)
            pred_tree.nodes[-1]["snvs"] = set()

        pred_tree.nodes[-1]["snvs"].update(snvs_lost_by_pred)

    pred_tree = remove_empty_nodes_from_tree(pred_tree)
    return pred_tree, true_tree


def remove_outlier_node_from_truth(true_tree):
    if true_tree.has_node(-1):
        true_tree.remove_node(-1)


def _update_tree_with_filtered_snv_set(filtered_mutations, tree):
    for node in tree.nodes:
        curr_snv_set = set(tree.nodes[node]["snvs"])
        new_snv_set = curr_snv_set.intersection(filtered_mutations)
        tree.nodes[node]["snvs"] = new_snv_set


def compute_clustering_metrics_from_trees(pred_tree, true_tree):
    pred_dict = get_clustering(pred_tree)

    true_dict = get_clustering(true_tree)

    result = compute_clustering_metrics(pred_dict, true_dict)
    return result


def compute_clustering_metrics(pred_dict, true_dict):
    pred = [pred_dict[x] for x in pred_dict]

    true = [true_dict[x] for x in pred_dict]

    result = dict(zip(["homogeneity", "completeness", "v_measure"], homogeneity_completeness_v_measure(true, pred)))

    result["num_muts_pred"] = len(pred_dict)

    result["num_muts_true"] = len(true_dict)

    return result


def compute_ancestor_descendant_metrics(pred_tree, true_tree):
    snvs = get_snvs(true_tree)

    pred_pairs = get_ancestor_descendant_pairs(pred_tree)

    true_pairs = get_ancestor_descendant_pairs(true_tree)

    result = {}

    for key, val in _pair_metrics(pred_pairs, true_pairs, snvs).items():
        result["ancestor_descendant_{}".format(key)] = val

    return result


def _pair_metrics(pred_pairs, true_pairs, snvs):
    tp = len(pred_pairs & true_pairs)

    fp = len(pred_pairs - true_pairs)

    fn, tn = _true_and_false_negatives(pred_pairs, snvs, true_pairs)

    result = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": safe_divide(tp, fn + tp),
        "fpr": safe_divide(fp, fp + tn),
        "specificity": safe_divide(tn, fp + tn),
        "precision": safe_divide(tp, fp + tp),
        "f_score": safe_divide(tp, tp + 0.5 * (fp + fn)),
    }

    return result


def _true_and_false_negatives(pred_pairs, snvs, true_pairs):
    all_pairs_iterator = itertools.permutations(snvs, 2)

    tn_counter = 0
    fn_counter = 0

    for pair in all_pairs_iterator:
        if (pair not in pred_pairs) and (pair not in true_pairs):
            tn_counter += 1
        if (pair not in pred_pairs) and (pair in true_pairs):
            fn_counter += 1

    return fn_counter, tn_counter


def safe_divide(x, y):
    if y == 0:
        return np.nan

    else:
        return x / y


def get_clustering(tree):
    result = {}

    for node in tree.nodes:
        for snv in tree.nodes[node]["snvs"]:
            result[snv] = node

    return result


def get_ancestor_descendant_pairs(tree):
    def get_desc_snvs(node, tree, p_snv):
        desc = []

        for child in tree.successors(node):
            if "lost_snvs" in tree.nodes[child]:
                if p_snv in tree.nodes[child]["lost_snvs"]:
                    continue
            desc.extend(tree.nodes[child]["snvs"])

            desc.extend(get_desc_snvs(child, tree, p_snv))

        return desc

    pairs = set()

    for node in tree.nodes:
        for p_snv in tree.nodes[node]["snvs"]:
            try:
                for c_snv in get_desc_snvs(node, tree, p_snv):
                    pairs.add((p_snv, c_snv))
            except RecursionError as e:
                print("Recursion limit hit, this is likely a cycle in the tree - which makes it no longer a tree.")
                return pairs

    return pairs


def get_snvs(tree):
    snvs = []

    for node in tree:
        snvs.extend(tree.nodes[node]["snvs"])

    return snvs


def remove_empty_nodes_from_tree(tree):
    empty_nodes = set()
    for node in tree.nodes:
        if len(tree.nodes[node]["snvs"]) == 0:
            empty_nodes.add(node)

    tree = _cleanup_tree(tree, empty_nodes)
    return tree


def _cleanup_tree(graph, excluded_clones):
    nodes = list(graph)

    for n in nodes:
        if n in excluded_clones:
            parent = next(graph.predecessors(n), None)
            if parent is None:
                graph.remove_node(n)
            else:
                graph = nx.contracted_nodes(graph, parent, n, self_loops=False)

    return graph


def read_pickle(file):
    with open(file, "rb") as f:
        loaded = pickle.load(f)
    return loaded
