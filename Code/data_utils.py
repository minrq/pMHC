import numpy as np
import torch
from config import ALLELE_LENGTH, PEP_LENGTH, AMINO_ACIDS

a2n_func = {amino:i+1 for i, amino in enumerate(AMINO_ACIDS)}
n2a_func = {i+1:amino for i, amino in enumerate(AMINO_ACIDS)}

def seq2num(sequences, allele):
    """ Convert sequences to interger values 
    example: "ACT" -> [1, 2, 3]
    """
    max_len = ALLELE_LENGTH if allele else PEP_LENGTH
    arrays = np.zeros((len(sequences), max_len))
    lengths = np.zeros((len(sequences)))

    for i, seq in enumerate(sequences):
        lengths[i] = len(seq)

        for j, amino in enumerate(seq):
            arrays[i,j] = a2n_func[amino]
    
    arrays = torch.LongTensor(arrays)
    return arrays

def num2seq(sequences, allele):
    new_sequences = []
    if allele:
        for i in range(sequences.shape[0]):
            seq = "".join([n2a_func[idx.item()] for idx in sequences[i,:]])
            new_sequences.append(seq)

    else:
        for i in range(sequences.shape[0]):
            seq = "".join([n2a_func[idx.item()] for idx in sequences[i,:] if idx.item() != 0])
            new_sequences.append(seq)

    return new_sequences
