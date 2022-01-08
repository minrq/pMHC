from mhcflurry import Class1PresentationPredictor
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from functools import partial
import argparse

import torch
import sys, os
sys.path.append("../vae/")
from vae2 import VAE	

from multiprocessing import Pool
import numpy as np
import pdb

def predict_score(allele, predictor, pep_vae, peptide_dict):
    peptide_vec = torch.tensor([peptide_dict[i] for i in range(len(peptide_dict))])
    
    peptide_seq = pep_vae.generate(peptide_vec.unsqueeze(0))
    
    score = predictor.predict(peptides = peptide_seq, alleles = [allele], verbose=0)
    
    return score['presentation_score'][0], peptide_seq[0]

def optimize_mhcflurry(threshold, pep_vae, rollouts, logpath, outpath, latent_size, allele):
    model = VAE(20, 64, latent_size, 'blosum')
    model.load_state_dict(torch.load(pep_vae, map_location='cpu'))
    
    predictor = Class1PresentationPredictor.load()
    
    func = partial(predict_score, allele, predictor, model)
    
    logger = JSONLogger(path="%s/log_%s.json" % (logpath, allele))

    bounds = {i: (-3, 3) for i in range(latent_size)}
    optimizer = BayesianOptimization(
        f=func,
        pbounds=bounds,
        random_state=1,
    )

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=100,
        n_iter=rollouts,
    )
    
    f = open("%s_%s.txt" % (outpath, allele), 'w')
    for i, res in enumerate(optimizer.res):
        score, peptide_seq = func(res['params'])

        #pdb.set_trace()        
        if score >= threshold:
            f.write("%s %s %.4f\n" % (allele, peptide_seq, score))

    f.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--alleles", type=str)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--vae_path", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--ncpu", type=int, default=20)
    args = parser.parse_args()

    alleles = [allele.strip() for allele in open(args.alleles, 'r').readlines()]
    
    #optimize_mhcflurry(args.vae_path, args.rollout, args.log_path, args.out_path, args.latent_size, alleles[0])
    with Pool(processes=args.ncpu) as pool:
        func = partial(optimize_mhcflurry, args.threshold, args.vae_path, args.rollout, args.log_path, args.out_path, args.latent_size)
       
        pool.map(func, alleles)
