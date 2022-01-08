from config import AMINO_ACIDS, BLOSUM
import numpy as np
import torch
import pdb

a2n_func = {amino:i+1 for i, amino in enumerate(AMINO_ACIDS)}
n2a_func = {i+1:amino for i, amino in enumerate(AMINO_ACIDS)}

blosum_dict = np.zeros((len(AMINO_ACIDS)+1, len(AMINO_ACIDS)))
for i, line in enumerate(open(BLOSUM, 'r').readlines()):
    if i == 0 or i == 21: continue
    elem = line.strip().split("\t")
    blosum_dict[i, :] = np.array(elem[1:-1]).astype(float)


onehot_dict = np.zeros((len(AMINO_ACIDS)+1, len(AMINO_ACIDS)))
for idx, amino in enumerate(AMINO_ACIDS):
    onehot_dict[idx+1, idx] = 1

def onehot(sequences):
    onehot_encodings = []
    for sequence in sequences:
        onehot_encodings.append(onehot_dict[sequence, :])

    onehot_encodings = np.stack(onehot_encodings, axis=0)
    return onehot_encodings

def blosum(sequences):
    blosum_encodings = []
    for sequence in sequences:
        blosum_encodings.append(blosum_dict[sequence, :])
    
    blosum_encodings = np.stack(blosum_encodings, axis=0)
    return blosum_encodings

def learned(amino):
    pass

def encode_amino(aminos, encode_method, learned=None):
    """ Encode sequences into embeddings
    """
    #lengths, seq_arrays = self._seq2num(sequences, allele)
    
    amino_encodings = torch.zeros((len(aminos), 0))

    aminos = ["".join(aminos)]
    seq_arrays = seq2vec(aminos)

    
    seq_encodings = np.zeros((len(aminos[0]), 0))

    if "deep" in encode_method:
        deep_encodings = deep(seq_arrays).squeeze(0)
        seq_encodings = np.concatenate((seq_encodings, deep_encodings), axis=1)
    
    if "blosum" in encode_method:
        blosum_encodings = blosum(seq_arrays).squeeze(0)
        seq_encodings = np.concatenate((seq_encodings, blosum_encodings), axis=1)

    if "onehot" in encode_method:
        onehot_encodings = onehot(seq_arrays).squeeze(0)
        seq_encodings = np.concatenate((seq_encodings, onehot_encodings), axis=1)
       
    return seq_encodings


def encode_sequence(sequences, encode_method, learned=None):
    #lengths, seq_arrays = self._seq2num(sequences, allele)
    
    seq_arrays = seq2vec(sequences)
    
    seq_encodings = np.zeros((len(sequences), max([len(seq) for seq in sequences]), 0))
    
    if "deep" in encode_method:
        deep_encodings = deep(seq_arrays)
        seq_encodings = np.concatenate((seq_encodings, deep_encodings), axis=2)
    
    if "blosum" in encode_method:
        blosum_encodings = blosum(seq_arrays)
        seq_encodings = np.concatenate((seq_encodings, blosum_encodings), axis=2)


    if "onehot" in encode_method:
        onehot_encodings = onehot(seq_arrays)
        seq_encodings = np.concatenate((seq_encodings, onehot_encodings), axis=2)

    return seq_encodings

def vec2seq(array):
    sequences = []
    for i in range(array.shape[0]):
        seq = "".join([n2a_func[idx.item()] for idx in array[i,:] if idx.item() != 0])
        sequences.append(seq)

    return sequences

def seq2vec(sequences):
    length = [len(seq) for seq in sequences]
    array  = np.zeros( (len(sequences), max(length)) )

    for i, seq in enumerate(sequences):
        for j, amino in enumerate(seq):
            array[i, j] = a2n_func[amino]

    return array.astype(int)

def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=1000, n_iter=10):

    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], \
                                   size=(n_warmup, bounds.shape[0]))

    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], \
                                   size=(n_iter, bounds.shape[0]))

    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max), \
                       x_try.reshape(1, -1), \
                       bounds=bounds, \
                       method="L-BFGS-B")

        if not res.success:
            continue

        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    return x_max

