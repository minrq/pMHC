import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/vae")
import bz2
from vae2 import VAE
from config import AMINO_ACIDS
from mhcflurry import Class1PresentationPredictor
from my_util import encode_sequence

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
import pdb
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from subproc_vec_env import SubprocMHCflurry
import time

from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"

class Predictor(nn.Module):
    def __init__(self, data_path, allele_path, vae_path, allele_size=680, input_size=32, times=5, batch_size=128, layer_size=8, positive_rate=0.2, sample_rate=0.5, ncpu=5, alleles=None, positive_buffer_size=5000):
        super().__init__()
        self.ncpu = ncpu
        
        #predictors[0] = Class1PresentationPredictor.load()
        #self.alleles = list(predictors[0].supported_alleles)
       
        self.vae_model = VAE(20, 64, 8, 'blosum', beta=0.001)
        self.vae_model.load_state_dict(torch.load(vae_path, map_location=device))

        #for param in self.vae_model.parameters():
        #    param.requires_grad=False
        
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.sample_rate = sample_rate    
        self.positive_rate = positive_rate
        
        self.peptides = []

        self.alleles_seq_dict = {}
        tmp_dict = {}
        for line in open(allele_path, 'r').readlines():
            elems = line.strip().split(" ")
            tmp_dict[elems[0]] = elems[1]

        seqs = []
        for allele in alleles:
            tmp_allele = allele.replace("*", "")
            if tmp_allele in tmp_dict and tmp_dict[tmp_allele] not in seqs:
                self.alleles_seq_dict[allele] = tmp_dict[tmp_allele]
                seqs.append(tmp_dict[tmp_allele])
        
        self.alleles = list(self.alleles_seq_dict.keys())
        
        #remove_idxs = []
        #for i, allele in enumerate(self.alleles):
        #    if allele not in self.allele_seqs:
        #        remove_idxs.append(i)
        #self.alleles = [allele for i, allele in enumerate(self.alleles) if i not in remove_idxs]
        
        dataset = bz2.BZ2File(data_path, 'r')
        
        for line in dataset:
            
            elems = line.decode('utf-8').split(",")
            if len(elems[1]) > 15 or len(elems[1]) < 8: continue
            if elems[0] not in self.alleles:
                other_name = elems[0].replace("*", "")
                if other_name not in self.alleles: continue
                self.peptides.append((other_name, elems[1]))
            else:    
                self.peptides.append((elems[0], elems[1]))
        
        self.peptides = list(self.peptides)
        self.allele_layer = nn.Sequential( 
                                nn.Linear(allele_size, times * input_size), 
                                nn.Tanh(),
                                nn.Linear(times * input_size,  input_size),
                            ).to(device)
        
        self.peptide_layer = nn.Sequential(
                                nn.Linear(layer_size, input_size),
                                nn.Tanh(),
                                nn.Linear(input_size, input_size),
                            ).to(device)
        
        self.hidden_layer = nn.Sequential( nn.ReLU(),
                                           nn.Linear(2 * input_size, input_size), 
                                           nn.ReLU(),
                                           nn.Linear(input_size, 1),
                                           nn.Sigmoid()).to(device)

        self.loss = nn.MSELoss()

        self.positive_pairs = deque(maxlen=positive_buffer_size)
        
         
    def sample(self, predictor):
        #rng = np.random.RandomState(0)
        #if len(self.positive_pairs) < 50:
        #    positive_rand_num = []
        #    rand_num = np.random.random((self.batch_size,))
        #    sample_iedb_num = len(np.where(rand_num >= self.sample_rate)[0])
        #    random_gen_num =  self.batch_size - sample_iedb_num
        #else:
        positive_rand_num = np.random.choice(len(self.positive_pairs), min(len(self.positive_pairs), int(self.batch_size * self.positive_rate)))
        rand_num = np.random.random((self.batch_size - len(positive_rand_num),))
        sample_iedb_num = len(np.where(rand_num >= self.sample_rate)[0])
        random_gen_num =  self.batch_size - len(positive_rand_num) - sample_iedb_num
        
        positive_alleles = [val[0] for i, val in enumerate(self.positive_pairs) if i in positive_rand_num]
        positive_peptides = [val[1] for i, val in enumerate(self.positive_pairs) if i in positive_rand_num]
        
        iedb_data_idxs = np.random.choice(len(self.peptides), sample_iedb_num)
        
        iedb_alleles = [self.peptides[idx][0] for idx in iedb_data_idxs]
        iedb_peptides = [self.peptides[idx][1] for idx in iedb_data_idxs]
        
        random_allele_idxs = np.random.choice(len(self.alleles), random_gen_num)
        random_alleles = [self.alleles[idx] for idx in random_allele_idxs]
        random_z = torch.randn( (random_gen_num, self.layer_size) ).to(device)
        random_peptides = self.vae_model.generate(random_z)
        
        alleles = positive_alleles + iedb_alleles + random_alleles
        peptides = positive_peptides + iedb_peptides + random_peptides
        
        targets = self.get_targets(predictor, alleles, peptides)
         
        return alleles, peptides, targets

    def get_targets(self, predictor, alleles, peptides):
        allele_names = {i:[allele] for i, allele in enumerate(alleles)}
        
        sample_names = [i for i in range(len(alleles))]
        
        batch_size = len(sample_names) // self.ncpu + 1
        batch_inputs = []
        
        for i in range(0, len(sample_names), batch_size):
            start_idx = i
            end_idx = min(len(sample_names), i + batch_size)

            batch_input = (peptides[start_idx:end_idx], {idx:allele_names[idx] for idx in range(start_idx, end_idx)}, [idx for idx in range(start_idx, end_idx)])
            batch_inputs.append(batch_input)
        
        targets = predictor.predict(batch_inputs)
        targets = torch.cat(targets, dim=0).squeeze().to(device)
        
        for i, target in enumerate(targets):
            if target.item() >= 0.5:
                self.positive_pairs.append( (alleles[i], peptides[i]) )
        
        return targets


    def forward(self, alleles, peptides, targets, trade_off=0.005):
        reconst_loss, amino_loss, kl_loss, acc, peptide_embeds = self.vae_model(peptides)
        
        peptide_embeds = self.peptide_layer(peptide_embeds)
        
        allele_seqs = [self.alleles_seq_dict[allele] for allele in alleles] 
        allele_vecs = torch.tensor(encode_sequence(allele_seqs, 'blosum')).float().transpose(1,2).flatten(1).to(device)
        
        alleles_embeds = self.allele_layer(allele_vecs)
        
        input_embeds = torch.cat(( peptide_embeds, alleles_embeds ), axis=1)
        
        output = self.hidden_layer(input_embeds).squeeze()
        
        prop_loss = self.loss(output, targets)
        loss = prop_loss + trade_off * reconst_loss
        return reconst_loss, prop_loss, loss, acc

    def optimize(self, peptide_num, allele, threshold=0.75, num_iter=20, lr=0.1):
        
        peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in peptide_len]
        
        peptide_embeds = torch.autograd.Variable(self.vae_model.encode(peptides), requires_grad=True).to(device)

        allele_embeds = self.allele_layer(allele)

        embeds = torch.cat(( peptide_embeds, allele_embeds ), axis=1)

        visited = []
        for step in range(num_iter):
            prop_val = self.hidden_layer(embeds)
            grad = torch.autograd.grad(prop_val, peptide_embeds)[0]

            peptide_embeds = peptide_embeds.data + lr * grad.data
            
            peptide_embeds = torch.autograd.Variable(peptide_embeds, requires_grad=True).to(device)
            if prop_val >= threshold:
                visited.append(peptide_embeds)


        peptide_seqs = self.vae_model.generate(visited)

        prediction = self.predictor.predict(peptides=peptide_seqs, alleles=[allele])

        return peptide_seqs, prediction

def load_mhcflurry():
    return Class1PresentationPredictor.load()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--vae_path", type=str, default="../vae/vae/evaluate19_step700000")
    parser.add_argument("--max_step", type=int, default=10000)
    parser.add_argument("--name", type=str, default="forward_model")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model_path", type=str, default="model")
    parser.add_argument("--peptide_path", type=str, default="../../data/mhcflurry/curated_training_data.with_mass_spec.csv.bz2")
    parser.add_argument("--allele_path", type=str, default="../../data/class1_pseudosequences.csv")
    parser.add_argument("--sample_rate", type=float, default=0.8)
    parser.add_argument("--positive_rate", type=float, default=0.2)
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--times", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--ncpu", type=int, default=10)
    
    parser.add_argument("--print_iter", type=int, default=20)
    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--stop_criteria", type=float, default=0.001)
    args = parser.parse_args()
    
    allele_size = 34 * 20 
    
    predictors = [load_mhcflurry for _ in range(args.ncpu)]
    para_mhcflurry = SubprocMHCflurry( predictors )
    
    alleles = list(load_mhcflurry().supported_alleles)

    prop = Predictor(args.peptide_path, args.allele_path, args.vae_path, allele_size, args.input_size, args.times, args.batch_size, args.hidden_size, ncpu=args.ncpu, alleles=alleles, positive_rate=args.positive_rate)
    
    optimizer = optim.Adam(prop.parameters(), lr=args.lr, amsgrad=True)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate) 
    
    reconst_losses, prop_losses, losses, accs = 0, 0, 0, 0
    t1 = time.time()


    #alleles, peptides = prop.sample(para_mhcflurry)
    for i in range(args.max_step):
        alleles, peptides, targets = prop.sample(para_mhcflurry)
        
        with torch.autograd.set_detect_anomaly(True):
            prop.zero_grad()
            
            reconst_loss, prop_loss, loss, acc = prop(alleles, peptides, targets)
            loss.backward()
            
            nn.utils.clip_grad_norm_(prop.parameters(), args.clip_norm)
            optimizer.step()
        
        reconst_losses = reconst_losses + float(reconst_loss.detach().cpu())
        prop_losses = prop_losses + float(prop_loss.detach().cpu())
        losses = losses + float(loss.detach().cpu())
        accs = accs + float(acc)
        #pdb.set_trace()
        if i > 0 and i % args.print_iter == 0:
            losses = losses / args.print_iter
            accs = accs/ args.print_iter
            prop_losses = prop_losses / args.print_iter
            reconst_losses = reconst_losses / args.print_iter
            t2 = time.time()
            print("iter: %d; timecost: %.4f; acc: %.4f; loss: %.4f; reconst_loss: %.4f; prop_loss: %.4f" % (i, t2 - t1, accs, losses, reconst_losses, prop_losses))
            sys.stdout.flush()
            t1 = t2
            if losses <= args.stop_criteria:
                break
            losses = 0
            accs = 0
            reconst_losses = 0
            prop_losses = 0

        if i > 0 and i % args.save_iter == 0:
            torch.save(prop.state_dict(), "./%s/%s_step%d.pt" % (args.model_path, args.name, i))

        if i > 0 and i % args.anneal_iter == 0:
            scheduler.step() 
            print("learning rate: %.6f" % scheduler.get_lr()[0])

    
    torch.save(prop.state_dict(), "./%s/model.pt" % (args.model_path))
