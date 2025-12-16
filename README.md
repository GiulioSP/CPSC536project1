PhyClone Early Stopping
=========
Phyclone is an algorithm for bayesian inference of clonal decomposition, designed for tumor subclone detection from bulk whole genome sequencing. This project will adapt Phyclone's source code to detect run completion early, aiming to save computational resources. 

A machine learning (ML) model checks for long plateaus in log likelihood within each replicate chain, suggesting that adequate sampling of the posterior distribution has occurred, so that chain is terminated early. After all chains finish, statistical tests evaluate the convergence of the run. 


Based on the paper: [PhyClone: Accurate Bayesian Reconstruction of Cancer Phylogenies from Bulk Sequencing](https://doi.org/10.1093/bioinformatics/btaf344)

Authors: Farbod Moghaddam, Giuli Sucar, Samuel Leung

--------

## Overview
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Benchmarking](#benchmarking)
-------

## Installation

Environments are managed by Conda, with one general environment for running ( `environment.yaml` ) and a specialized environment for calculating benchmarking metrics ( `benchmarking/metrics_environment.yaml` ).

Create a local conda environment from each .yaml file.
```
conda env create -f environment.yaml -n phyclone_custom 
conda env create -f benchmarking/metrics_environment.yaml -n metrics 
```

Install project locally with pip.    
Add `-e` option for live editing.
```
pip install .
```

Activate the conda environment for running the adapted phyclone.
```
conda activate phyclone_custom
```

-----------------

## Basic Usage

### Input

Refer to the [PhyClone repository](https://github.com/Roth-Lab/PhyClone) version 0.7.1 for a thorough description of usage and functionality. 

The only change for input options from version 0.7.1 is an added option `--plateau-iterations` (`-pl` for short) that sets the number of iterations that the ML model check for plateaus. This number also serves as the ammount of iterations desired after a plateau to sufficiently sample the posterior, and should be larger for more complex runs. 
The minimum (and default) value is 20 and the maximum is 1024. 

Example command:
```
phyclone run -i INPUT.tsv -c CLUSTERS.tsv -o TRACE.pkl.gz -pl 200 --num-chains 4
``` 

### Output

Outputs include typical phyclone outputs and convergence reports based on r̂ and ESS statistical tests across all chains. 


## Benchmarking

Benchmarking scripts of the original PhyClone publication were not published, but were well described and kindly provided by the first author Emilia Hurtado, then adapted for this context. Scripts and results are in `benchmarking/` directory.

Benchmarking metrics requires 2 sets of files, one source of truth and one from the run being evaluated. For each set, two output files are needed: a newick tree ( `TREE.nwk` ) and a results table ( `TABLE.tsv` ).

Possible steps are:
1. Activate the conda environment for benchmarking
   ```
   conda activate metrics
   ```
2. Obtain `TREE.nwk` and `TABLE.tsv` outputs from a phyclone run with MAP (*maximum a posteriori* point estimate) or consensus (summary of all samples). 
   ```
   phyclone map -i TRACE.pkl.gz -t TREE.nwk -o TABLE.tsv
   ```
   or 
   ```
   phyclone consensus -i TRACE.pkl.gz -t TREE.nwk -o TABLE.tsv
   ```
3. Process the output into a networkx graph compressed in pickle format ( `GRAPH.pkl` ).
   ```
   python post_process_phyclone.py -i TABLE.tsv -t TREE.nwk -o GRAPH.pkl
   ```
4. Also process the source of truth into a networkx graph.
   ```
   python post_process_phyclone.py -i TRUTH_TABLE.tsv -t TRUTH_TREE.nwk -o TRUTH_GRAPH.pkl
   ```
5. Compute metrics comparing run outputs with source of truth. Note that `PROJECT_NAME`, `DEPTH`, `NUM_MUTATIONS` and `NUM_SAMPLES` are necessary inputs but are only used for reporting purposes and are not used in any metric calculation. 
   ```
   python compute_metrics.py -p GRAPH.pkl -t TRUTH_GRAPH.pkl -o METRICS_OUT_PATH.tsv -P PROJECT_NAME -D DEPTH -M NUM_MUTATIONS -S NUM_SAMPLES
   ```

# License

This project is licensed under the GNU General Public License v3 or later (GPLv3+), see the [LICENSE](LICENSE.md) file for details.
