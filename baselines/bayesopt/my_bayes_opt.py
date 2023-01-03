from mhcflurry import Class1PresentationPredictor
from functools import partial
import argparse
import time
import pdb
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf

import sys, os
sys.path.append("../vae/")
from vae2 import VAE	
from multiprocessing import Pool
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_score(allele, predictor, pep_vae, peptide_dict):
    peptide_vec = torch.tensor([peptide_dict[i] for i in range(len(peptide_dict))])
    
    peptide_seq = pep_vae.generate(peptide_vec.unsqueeze(0))
    
    score = predictor.predict(peptides = peptide_seq, alleles = [allele], verbose=0)
    
    return score['presentation_score'][0], peptide_seq[0]

def get_initial_data(threshold, allele, init_points, predictor, vae_model, bounds, latent_size=8):
    #print(";afjioawej %d %d" % (init_points, latent_size))
    train_x = unnormalize(torch.rand((init_points, latent_size), device=device).float(), bounds=bounds)
    random_peptides = vae_model.generate(train_x)
    
    targets = predictor.predict(peptides = random_peptides, alleles=[allele], verbose=0)['presentation_score']
    
    result = []
    for i, target in enumerate(targets):
        if target >= threshold:
            result.append( "%s %s %.4f\n" % (allele, random_peptides[i], target) )
    
    train_y = torch.tensor(targets.to_numpy(), device=device).float().unsqueeze(-1)
    
    return train_x, train_y, result

def init_and_fit_model(train_x, train_obj, state_dict=None):
    """ Model fitting helper function """
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    
    fit_gpytorch_model(mll)
    return mll, model

def optimize_mhcflurry(threshold, pep_vae, rollouts, steps, init_points, logpath, \
                       outpath, latent_size, max_size, mode, hour, \
                       allele, BATCH_SIZE=1, NUM_STARTS=5, RAW_SAMPLES=200):
    st_time = time.time()
    model = VAE(20, 64, latent_size, 'blosum')
    model.load_state_dict(torch.load(pep_vae, map_location='cpu'))
    
    bounds = torch.tensor([[-6.0] * latent_size, [6.0] * latent_size], device=device).float()
    #pdb.set_trace()
    predictor = Class1PresentationPredictor.load()
    
    func = partial(predict_score, allele, predictor, model)
    
    results = []
    
    for i in range(rollouts):
        train_x, train_y, result = get_initial_data(threshold, allele, init_points, predictor, model, bounds, latent_size=latent_size)

        #results.update(result)
        
        state_dict, next_peptide, next_y = None, None, None
        best_peptide, best_y = None, 0
        
        result = []
        for j in range(steps):
                
            try:
                _, gp_model = init_and_fit_model(
                           normalize(train_x, bounds=bounds),
                           standardize(train_y),
                           state_dict=state_dict,)
            except Exception as e:
                print(e)
                break
            
            state_dict = gp_model.state_dict()

            UCB = UpperConfidenceBound(gp_model, beta=0.1)
            
            next_x, _ = optimize_acqf(
                            acq_function=UCB,
                            bounds=bounds,
                            q=BATCH_SIZE,
                            num_restarts=NUM_STARTS,
                            raw_samples=RAW_SAMPLES)
           
            #next_x = unnormalize(next_x.detach(), bounds=bounds)
            next_peptide = model.generate(next_x)
            next_y = predictor.predict(peptides=next_peptide, alleles=[allele], verbose=0)['presentation_score'].to_numpy()
            next_y = torch.tensor(next_y, device=device).float().to(device).unsqueeze(-1)
            pdb.set_trace()
            is_break = False
            for i, y in enumerate(next_y):
                if mode < 2 and y >= threshold:
                    is_break = True
                    result.append("%s %s %.4f\n" % (allele, next_peptide[i], y) )
                    break
                    
                if y > best_y: 
                    best_y = y
                    best_peptide = next_peptide[i]
                
            if is_break and mode == 1: break
            
            ck_time = time.time()
            
            if ck_time - st_time >= hour * 3600: break         
            train_x = torch.cat( (train_x, next_x) )
            train_y = torch.cat( (train_y, next_y) )
        
        if mode == 2 and next_y is not None:
            results.append( "%s %s %.4f\n" % (allele, next_peptide[0], next_y[0]) )
        elif mode == 1 and len(result) == 0:
            results.append( "%s %s %.4f\n" % (allele, best_peptide, best_y) )
        else:
            results.extend(result)
        
        if time.time() - st_time >= hour * 3600: break
        if len(results) >= max_size: break
    
    end_time = time.time()
    results.append("time cost: %.4f\n" % (end_time - st_time))
    
    return results
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--alleles", type=str)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--vae_path", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--init_points", type=int, default=100)
    parser.add_argument("--opt_step", type=int, default=5)

    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--hour", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=500)
    parser.add_argument("--ncpu", type=int, default=15)
    args = parser.parse_args()

    alleles = [allele.strip() for allele in open(args.alleles, 'r').readlines()]
    
    t1 = time.time()
    optimize_mhcflurry(args.threshold, args.vae_path, args.rollout, args.opt_step, args.init_points, args.log_path, args.out_path, args.latent_size, args.max_size, args.mode, args.hour, alleles[0])
    #with Pool(processes=args.ncpu) as pool:
    #    func = partial(optimize_mhcflurry, args.threshold, args.vae_path, args.rollout, \
    #                   args.opt_step, args.init_points, args.log_path, args.out_path, \
    #                   args.latent_size, args.max_size, args.mode, args.hour)
    #   
    #    result = pool.map(func, alleles)

    result = [res for tmp in result for res in tmp]
    f = open(args.out_path, 'w')
    for string in result:
        f.write(string)

    f.close()
    print("time cost: %.4f" % (time.time() - t1))
