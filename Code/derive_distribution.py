from mhcflurry import Class1PresentationPredictor
import pdb
import bz2
import sys, os
from config import AMINO_ACIDS
import numpy as np
import pickle

def calculate_distribution(peptides, length, amino_dict=None):
    dists = np.zeros((length, 20))
    
    for peptide in peptides:
        for i, amino in enumerate(peptide):
            dists[i, amino_dict[amino]] += 1

    dists = dists / len(peptides)
    return dists
        

allele_sequences = {}
for line in open("../class1_pseudosequences.csv", 'r').readlines()[1:]:
    elems = line.strip().split(" ")
    allele_sequences[elems[0]] = elems[1]

dataset = open("../mhcflurry/Data_S3.csv", 'r')
positive_dicts = {}

allele_dict = {}
i = 0
for line in dataset:
    if i == 0:
        i += 1
        continue
    elems = line.split(",")
    if elems[5] != "mass_spec" or elems[3] == ">": continue
    if float(elems[2]) < 500 or (float(elems[2]) == 500 and elems[3] != "="):
        if "X" in elems[1]: continue
        if elems[0] not in allele_sequences:
            cand = elems[0].replace("*", "")
            
            if cand in allele_sequences:
                allele_sequence = allele_sequences[cand]
            else:
                cand = cand.replace(":","")
                if cand in allele_sequences:
                    allele_sequence = allele_sequences[cand]
                else:
                    print("not match %s" % elems[0])
                    continue
        else:
            allele_sequence = allele_sequences[elems[0]]
        
        allele_dict[allele_sequence] = elems[0]
        if allele_sequence in positive_dicts:
            if len(elems[1]) in positive_dicts[allele_sequence]:
                positive_dicts[allele_sequence][len(elems[1])].append(elems[1])
            else:
                positive_dicts[allele_sequence][len(elems[1])] = [elems[1]]
        else:
            positive_dicts[allele_sequence] = {}
            positive_dicts[allele_sequence][len(elems[1])] = [elems[1]]

amino_dict = {}
for i, amino in enumerate(AMINO_ACIDS): amino_dict[amino] = i

positive_sequences = list(positive_dicts.keys())
dist_dict = {}
for sequence in positive_sequences:
    for length in range(8, 16):
        if length in positive_dicts[sequence]:
            dist = calculate_distribution(positive_dicts[sequence][length], length, amino_dict=amino_dict)
            
            if len(np.nonzero(dist)[0])/(length * 20) > 0.6 and len(positive_dicts[sequence][length]) >= 20:
                if sequence not in dist_dict: dist_dict[sequence] = {}
                
                dist_dict[sequence][length] = dist

with open("../dict3.pkl", 'wb') as f:
    pickle.dump(dist_dict, f, pickle.HIGHEST_PROTOCOL)


