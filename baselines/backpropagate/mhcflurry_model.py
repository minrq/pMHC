from mhcflurry import Class1PresentationPredictor
from mhcflurry.flanking_encoding import FlankingEncoding

import pandas as pd
import numpy as np

def read_data(path):
    df = pd.read_csv(path)

    train_df = df.loc[df.sample_id.isin(sorted(df.sample_id.unique())[:3])]

    test_df = df.loc[~df.sample_id.isin(train_df.sample_id.unique())]

    test_df = test_df.sample(frac=0.01, weights=test_df.hit + 0.01)

    experiment_to_alleles = (df.drop_duplicates("sample_id").set_index("sample_id").hla.str.split().to_dict())

    return train_df, test_df, experiment_to_alleles


def train(train_df, experiment_to_alleles):
    affinity_predictor = Class1AffinityPredictor()
    affinity_predictor.fit_class1_pan_allele_models(
        n_models=1, 
        architecture_hyperparameters_list=[hyperparameters], 
        alleles=train_df.alleles.values,
        peptides=train_df.peptides.values)
    affinity_out = affinity_predictor.output
    
    processing_predictor = Class1ProcessingPredictor()
    processing_predictor.fit(
        sequences=FlankingEncoding(
            peptides=train_df.peptide.values,
            n_flanks=train_df.n_flank.values,
            c_flanks=train_df.c_flank.values),
        targets=train_df.hit.values)
    processing_out = processing_predictor.output
    
    predictor = Class1PresentationPredictor(
        affinity_predictor =affinity_predictor,
        processing_predictor_without_flanks=processing_predictor,
        processing_predictor_with_flanks=None)

    predictor.fit(
        targets=train_df.hit.values,
        peptides=train_df.peptide.values,
        sample_names=train_df.sample_id.values,
        alleles=experiment_to_alleles,
        n_flanks=train_df.n_flank.values,
        c_flanks=train_df.c_flank.values,
        verbose=2)

    x = concatenate([affinity_out, processing_out])
    x = predictor.weights_dataframe.loc["without_flanks"]

    model=Model(inputs=[affinity_predictor.input, processing_predictor.input], outputs=x)
    
    x_dict = {'peptides': ["SIINFEKL","RELYLNGPEPESS","SIINFEKLQ"], \
              'alleles': ['HLA-A*02:01', 'HLA-A*03:01']}

    model.predict([x_dict, x_dict]) 
    model.predict([x_dict, x_dict])

    return model

def optimize(test_df, predictor):
    peptides=test_df.peptides.values.copy()
    loss = K.mean(predictor.output[:, 1])

    grads = K.gradients(loss, predictor.layers[1].output)[0]


if __name__ == "__main__":
    path = "../../data/mhcflurry/curated_training_data.with_mass_spec.csv.bz2"

    train_df, test_df = read_data(path)
