"""
Model select class1 pan-allele models.

APPROACH: For each training fold, we select at least min and at most max models
(where min and max are set by the --{min/max}-models-per-fold argument) using a
step-up (forward) selection procedure. The final ensemble is the union of all
selected models across all folds.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import hashlib
from pprint import pprint

import numpy
import pandas
import pdb
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

sys.path.append("/users/PES0781/ziqichen/anaconda3/envs/mhcflurry/lib/python3.6/site-packages/mhcflurry/")

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import configure_logging
from mhcflurry.local_parallelism import (
    worker_pool_with_gpu_assignments_from_args,
    add_local_parallelism_args)
from mhcflurry.cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)
from mhcflurry.regression_target import from_ic50
import sklearn

# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=False,
    help=(
        "Model selection data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read models")
parser.add_argument(
    "--out",
    required=True,
    type=str,
    help="Directory to write selected models")
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write selected models")
parser.add_argument(
    "--min-models-per-fold",
    type=int,
    default=2,
    metavar="N",
    help="Min number of models to select per fold")
parser.add_argument(
    "--max-models-per-fold",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per fold")
parser.add_argument(
    "--test_alleles",
    type=str,
    required=True,
    help="Directory to write selected models")
parser.add_argument(
    "--mass-spec-regex",
    metavar="REGEX",
    default="mass[- ]spec",
    help="Regular expression for mass-spec data. Runs on measurement_source col."
    "Default: %(default)s.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)

def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.out_models_dir = os.path.abspath(args.out_models_dir)

    configure_logging(verbose=args.verbosity > 1)

    df = pandas.read_csv(args.data)
    print("Loaded data: %s" % (str(df.shape)))

    input_predictor = Class1AffinityPredictor.load(
        args.models_dir, optimization_level=0)
    print("Loaded: %s" % input_predictor)
    
    alleles = input_predictor.supported_alleles
    (min_peptide_length, max_peptide_length) = (
        input_predictor.supported_peptide_lengths)

    metadata_dfs = {}
    
    fold_cols = [c for c in df if c.startswith("fold_")]
    num_folds = len(fold_cols)
    
    if num_folds <= 1:
        raise ValueError("Too few folds: ", num_folds)

    df = df.loc[
        (df.peptide.str.len() >= min_peptide_length) &
        (df.peptide.str.len() <= max_peptide_length)
    ]
    
    print("Subselected to %d-%dmers: %s" % (
        min_peptide_length, max_peptide_length, str(df.shape)))
    
    print("Num folds: ", num_folds, "fraction included:")
    print(df[fold_cols].mean())
    
    # Allele names in data are assumed to be already normalized.
    df = df.loc[df.allele.isin(alleles)]
    print("Subselected to supported alleles: %s" % str(df.shape))
    
    metadata_dfs["model_selection_data"] = df

    df["mass_spec"] = df.measurement_source.str.contains(
        args.mass_spec_regex)

    def make_train_peptide_hash(sub_df):
        train_peptide_hash = hashlib.sha1()
        for peptide in sorted(sub_df.peptide.values):
            train_peptide_hash.update(peptide.encode())
        return train_peptide_hash.hexdigest()
    
    folds_to_predictors = dict(
        (int(col.split("_")[-1]), (
            [],
            make_train_peptide_hash(df.loc[df[col] == 1])))
        for col in fold_cols)
    
    
    for model in input_predictor.class1_pan_allele_models:
        training_info = model.fit_info[-1]['training_info']
        fold_num = training_info['fold_num']
        assert num_folds == training_info['num_folds']
        (lst, hash) = folds_to_predictors[fold_num]
        train_peptide_hash = training_info['train_peptide_hash']    
        #numpy.testing.assert_equal(hash, train_peptide_hash)
        lst.append(model)
        
    
    print(folds_to_predictors)
    work_items = []
    for (fold_num, (models, _)) in folds_to_predictors.items():
        work_items.append({
            'fold_num': fold_num,
            'models': models,
            'min_models': args.min_models_per_fold,
            'max_models': args.max_models_per_fold,
        })
    
    GLOBAL_DATA["data"] = df
    GLOBAL_DATA["input_predictor"] = input_predictor
     
    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    result_predictor = Class1AffinityPredictor(
        allele_to_sequence=input_predictor.allele_to_sequence,
        metadata_dataframes=metadata_dfs)
    
    serial_run = not args.cluster_parallelism and args.num_jobs == 0
    worker_pool = None
    start = time.time()
    
    
    full_data = GLOBAL_DATA["data"]
    full_data = full_data.loc[(~full_data.fold_0) | (~full_data.fold_1) | (~full_data.fold_2) | (~full_data.fold_3), :].copy()
    input_predictor = GLOBAL_DATA["input_predictor"]
    
    test_alleles = [line.strip() for line in open(args.test_alleles, 'r').readlines()]
    all_predictions_df = full_data.copy()
    
    for i, allele in enumerate(test_alleles):
        allele_df = full_data.loc[(full_data['allele'] == allele) & (full_data['measurement_source'] != "pseudo_label")].copy()
        
        for fold in range(4):
            predictions_df = allele_df.loc[allele_df["fold_%d" % fold] == 0].copy()
            
            peptides = EncodableSequences.create(predictions_df.peptide.values)
            alleles = AlleleEncoding(
                predictions_df.allele.values,
                borrow_from=input_predictor.master_allele_encoding)
            
            #if predictions_df['fold_%d' % fold].any(): pdb.set_trace()
            
            for (i, model) in enumerate(folds_to_predictors[fold][0]):
                predictions_df.loc[:, i] = model.predict(peptides, alleles)
                if predictions_df['fold_%d' % fold].any(): pdb.set_trace()
                
                
                all_predictions_df.loc[(all_predictions_df['allele'] == allele) & \
                                       (~all_predictions_df['fold_%d' % fold]), i] = predictions_df[i].copy()
    
    all_predictions_df.to_csv(args.out)
    

if __name__ == '__main__':
    run()
