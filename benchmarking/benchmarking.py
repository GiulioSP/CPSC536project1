from sklearn.metrics.cluster import v_measure_score
import pandas as pd

def v_measure(): #(true_labels, pred_labels):

    print("trying")

    true_path = '/home/giuliosp/Desktop/Joy_Lab/CPSC536/CPSC536project1/testing/test1cons.tsv'
    true_df = pd.read_csv(true_path, sep='\t')

    pred_path = '/home/giuliosp/Desktop/Joy_Lab/CPSC536/CPSC536project1/testing/test1map.tsv'
    pred_df = pd.read_csv(pred_path, sep='\t')


    # only sort by mutation_id and keep shared mutations (no merge)
    common = pd.Index(true_df['mutation_id']).intersection(pred_df['mutation_id'])
    true_sorted = true_df[true_df['mutation_id'].isin(common)].sort_values('mutation_id').reset_index(drop=True)
    pred_sorted = pred_df[pred_df['mutation_id'].isin(common)].sort_values('mutation_id').reset_index(drop=True)

    #aligned = pd.DataFrame({
    #    'mutation_id': true_sorted['mutation_id'].values,
    #    'clone_id_true': true_sorted['clone_id'].values,
    #    'clone_id_pred': pred_sorted['clone_id'].values
    #})
    #aligned.to_csv('/home/giuliosp/Desktop/Joy_Lab/CPSC536/CPSC536project1/testing/test1intersection.tsv', sep='\t', index=False)

    true_labels = aligned['clone_id_true'].to_numpy()
    pred_labels = aligned['clone_id_pred'].to_numpy()
    v = v_measure_score(true_labels, pred_labels)
    print("V-measure:", str(v))
    return v

v_measure()


