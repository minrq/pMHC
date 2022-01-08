import argparse
import torch
from mhcflurry_vae6 import Predictor
#from config import data_path, allele_path, vae_path
from multiprocessing import Pool
from functools import partial
import pdb
import copy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/vae")
from my_util import encode_sequence
from config import AMINO_ACIDS, LENGTH_DIST
import numpy as np
from mhcflurry import Class1PresentationPredictor
import time

vae_path = "/users/PES0781/ziqichen/peptideproject/Code/RL/baseline/vae/vae/evaluate19_step700000"
data_path = "/users/PES0781/ziqichen/peptideproject/Code/RL/data/mhcflurry/curated_training_data.with_mass_spec.csv.bz2"
allele_path = "/users/PES0781/ziqichen/peptideproject/Code/RL/data/class1_pseudosequences.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

def optimize(model_path, lr, num, num_iter, latent_size, max_size, hour, mode, allele, threshold=0.75):
    st_time = time.time()
    
    model = Predictor(data_path, allele_path, vae_path, alleles=[allele])
    model.load_state_dict( torch.load(model_path, map_location=device) )

    pdb.set_trace()
    predictor = Class1PresentationPredictor.load()
    predict_seqs = []
     
    #peptide_vec = torch.rand(latent_size)
    if allele not in model.alleles_seq_dict:
        allele = allele.replace("*", "") #.replace(":", "")
        if allele not in model.alleles_seq_dict:
            allele = allele.replace(":", "")
            if allele not in model.alleles_seq_dict:
                print("wrong")
                return []
    allele_vec = torch.tensor(encode_sequence([model.alleles_seq_dict[allele]], 'blosum')).float().transpose(1,2).flatten(1).to(device)
    
    allele_embed = model.allele_layer(allele_vec)
    
    visited = set()
    qualified = set()
    results = []
    for i in range(num):
        ck_time = time.time()
        if ck_time - st_time >= hour * 3600: break
        
        newly_added = set()
        random_peptide_len = np.random.choice(np.arange(8, 16), 1, p=LENGTH_DIST)
        random_peptide = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in random_peptide_len][0]

        peptide_embed = torch.autograd.Variable(model.vae_model.sample(model.vae_model.encode([random_peptide]))[0], requires_grad=True).to(device)
        
        for step in range(num_iter):
            peptide_embed_ = model.peptide_layer(peptide_embed)
            embed = torch.cat(( peptide_embed_, allele_embed ), axis=1)
            prop_val = model.hidden_layer(embed).squeeze()
            
            if mode == 0 and prop_val >= threshold:
                peptide_seq = model.vae_model.generate(peptide_embed)[0]
                
                newly_added.add( (prop_val, peptide_seq) )
            
            grad = torch.autograd.grad(prop_val, peptide_embed)[0]
            
            new_peptide_embed = peptide_embed.data + lr * grad.data
            
            peptide_embed = torch.autograd.Variable(new_peptide_embed, requires_grad=True).to(device)
        
        if mode != 0:
            peptide_seq = model.vae_model.generate(peptide_embed)[0]
            newly_added.add( (0, peptide_seq) )
        
        del peptide_embed, grad
    
        if len(newly_added) > 0:
            unvisited = [tmp[1] for tmp in list(newly_added) if tmp not in visited]
            prediction = predictor.predict(peptides=unvisited, alleles=[allele], verbose=0)
            
            for i, seq in enumerate(unvisited):
                #if prediction['presentation_score'][i] < threshold: continue
         
                s = "%s %s %.4f\n" % (allele, seq, prediction['presentation_score'][i])
                results.append(s)

                if len(results) >= max_size: break

        if len(results) >= max_size: break

    end_time = time.time()
    s = "time cost: %.4f\n" % (end_time - st_time)
    results.append(s)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alleles", type=str, default="../mcts/alleles.txt")
    parser.add_argument("--num", type=int, default=2000)
    parser.add_argument("--num_iter", type=int, default=50)
    parser.add_argument("--out", type=str, default="./out.txt")
    parser.add_argument("--lr", type=float, default=5)
    parser.add_argument("--ncpu", type=int, default=5)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--hour", type=int, default=5)
    parser.add_argument("--model_path", type=str, default="./model1/model.pt")
    parser.add_argument("--max_size", type=int, default=10000)
    parser.add_argument("--mode", type=int, default=0)
    
    args = parser.parse_args()
    
    alleles = [allele.strip() for allele in open(args.alleles, 'r').readlines()]
   
    
    result = [optimize(args.model_path, args.lr, args.num, args.num_iter, args.latent_size, args.max_size, args.hour, args.mode, alleles[0])]
    pdb.set_trace()
    with Pool(processes=args.ncpu) as pool:
        func = partial(optimize, args.model_path, args.lr, args.num, args.num_iter, \
                       args.latent_size, args.max_size, args.hour, args.mode)
       
        result = pool.map(func, alleles)

    result = [r for tmp in result for r in tmp]
    out_file = open(args.out, 'w')
    for r in result:
        out_file.write(r)
    out_file.close()
