import argparse
from functools import partial
import numpy as np
import random
import pickle

import pdb
import random
import math
import numpy as np
from mhcflurry import Class1PresentationPredictor
import multiprocessing as mp
from multiprocessing import Pool

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
C_PUCT = 0.5

class Predictor():
    def __init__(self, allele):
        self.allele = allele

        self.predictor = Class1PresentationPredictor.load()
        
    def predict(self, peptides):
        
        predictions = self.predictor.predict(peptides=peptides, alleles=[self.allele], verbose=0)

        return predictions['presentation_score']

class MCTSNode():

    def __init__(self, seqs, W=0, N=0, P=0):
        self.seqs = seqs
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT *  math.sqrt(2 * n / ( 1+self.N ))


def mcts_rollout(node, state_map, max_len, min_len, predictor):
    seqs = node.seqs
    if len(seqs) >= max_len:
        return node.P
    
    # Expand if this node has never been visited
    if len(node.children) == 0:
        for amino in AMINO_ACIDS:
            new_seqs = seqs + amino
            #print('new_smiles', node.smiles, '->', new_smiles)
            if new_seqs in state_map:
                new_node = state_map[new_seqs] # merge identical states
            else:
                new_node = MCTSNode(new_seqs)
            
            node.children.append(new_node)
        
        state_map[seqs] = node
        
        if len(node.children[0].seqs) >= min_len:
            scores = predictor.predict([x.seqs for x in node.children])
            for child, score in zip(node.children, scores):
                child.P = score
        
    sum_count = sum([c.N for c in node.children])
    random.shuffle(node.children)
    selected_node = max(node.children, key=lambda x : x.Q() + x.U(sum_count))
    
    v = mcts_rollout(selected_node, state_map, max_len, min_len, predictor)
    
    selected_node.W += v
    selected_node.N += 1
    return max(v, selected_node.P)

def mcts(allele, n_rollout, max_len, min_len, prop_delta):
    predictor = Predictor(allele)
    
    root = MCTSNode( "" ) 
    state_map = {"" : root}
    try:
        for _ in range(n_rollout):
            mcts_rollout(root, state_map, max_len, min_len, predictor)
    except:
        return []
    nodes = [node for _,node in state_map.items() if len(node.seqs) >= min_len and node.P >= prop_delta]
    
    return nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alleles", type=str)
    parser.add_argument("--num_rollout", type=int)
    parser.add_argument("--out", type=str)
    parser.add_argument("--prop_delta", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--ncpu", type=int, default=mp.cpu_count())

    args = parser.parse_args()
    
    alleles = [allele.strip() for allele in open(args.alleles, 'r').readlines()]
    
    work_func = partial(mcts, 
                        n_rollout=args.num_rollout, 
                        prop_delta=args.prop_delta, 
                        max_len=args.max_len, 
                        min_len=args.min_len)

    with Pool(processes=args.ncpu) as pool:
        results = pool.map(work_func, alleles)
    
    f = open(args.out, 'w')
    for i, result in enumerate(results):
        for node in result:
            f.write("%s %s %.4f\n" % (alleles[i], node.seqs, node.P))
    f.close()
