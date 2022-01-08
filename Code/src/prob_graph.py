import numpy as np
from config import BLOSUM, AMINO_ACIDS, PATH
import pickle

class ProbGraph:
    def __init__(self, device):
        alleles = [allele.strip() for allele in open(PATH + "/data/alleles.txt", 'r')]
        
        self.alleles = alleles
        
        self._build_blosum_dict()
        self.get_similar_map()
        self.device = device

        self.pos_dict = {}
        self.amino_dict = {}
         
    def _build_blosum_dict(self):
        dict_file = open(BLOSUM, 'r')
        
        blosum_dict = np.zeros((len(AMINO_ACIDS), len(AMINO_ACIDS)))
        blosum_map  = {}
        for i, line in enumerate(dict_file.readlines()):
            if i == 0 or i == 21: continue
            elem = line.strip().split("\t")
            blosum_map[AMINO_ACIDS[int(elem[0])]] = i-1
            blosum_dict[i-1, :] = np.array(elem[1:-1]).astype(float)

        self.blosum_map = blosum_map
        self.blosum_dict = blosum_dict

    def find_similar_alleles(self, sequence, length):
        sequences = [key for key, value in self.dist.items() if length in value]

        if len(sequences) == 0: return None, -100
        dist = np.ones(len(sequences)) * -1000

        for i, cand in enumerate(sequences):
            cost = 0
            for j, amino in enumerate(sequence):
                try:
                    cost += self.blosum_dict[self.blosum_map[amino],self.blosum_map[cand[j]]]
                except:
                    print("wrong cost")
                    pdb.set_trace()
            dist[i] = cost
        
        cost = np.max(dist)
        if cost < 110: return None, -100
        
        similar_sequence = sequences[np.argmax(dist)]
        return similar_sequence, np.max(dist)

    def get_similar_map(self):
        self.map_dict = {}
        
        #mapping_sequences = [sequence for sequence in self.alleles if sequence not in length_dist]
        
        for sequence in self.alleles:
            for length in range(8, 16):
                if sequence not in self.dist or length not in self.dist[sequence]:
                    
                    mapped_sequence, cost = self.find_similar_alleles(sequence, length)
                    
                    if sequence in self.map_dict:
                        self.map_dict[sequence][length] = mapped_sequence
                    else:
                        self.map_dict[sequence] = {}
                        self.map_dict[sequence][length] = mapped_sequence

        for seq in self.map_dict:
            uncovered_length = []
            for length in self.map_dict[seq]:
                mapped_seq = self.map_dict[seq][length]
                if mapped_seq is not None:
                    if seq not in self.dist: self.dist[seq] = {}
                    
                    self.dist[seq][length] = self.dist[mapped_seq][length]

                else:
                    uncovered_length.append(length)

            # unmatched alleles
            if 0 < len(uncovered_length) < 8:
                for length in uncovered_length:
                    pad = np.ones((length - 8, 20)) / len(AMINO_ACIDS)
                    dist = np.mean([np.concatenate([tmp[:4, :], pad, tmp[-4:, :]], axis=0) for length, tmp in self.dist[seq].items()], axis=0)
                    self.dist[seq][length] = dist
                    
            elif len(uncovered_length) == 8:
                self.dist[seq] = {}
                for length in uncovered_length:
                    self.dist[seq][length] = np.ones((length, 20)) * -1
    
    def get_dist(self, sequence, length):
        #if sequence in self.dist and length in self.dist[sequence]:
        return self.dist[sequence][length]
        #
        #mapped_sequence = self.map_dict[sequence][length]
        #
        #if mapped_sequence is None: return np.ones(len(AMINO_ACIDS), len(AMINO_ACIDS)) / len(AMINO_ACIDS)
        #return self.dist[mapped_sequence][length]

    def get_pos(self, sequence, length):
        if sequence in self.pos_dict and length in self.pos_dict[sequence]:
            return self.pos_dict[sequence][length]
        
        dist = self.get_dist(sequence, length)
        if np.min(dist) == -1: return np.zeros((length))
        
        pos_dist = pos_entropy(dist)
        
        if sequence not in self.pos_dict: self.pos_dict[sequence] = {}
        
        self.pos_dict[sequence][length] = pos_dist
        return pos_dist

    def get_amino(self, sequence, length, pos):
        if sequence in self.amino_dict:
            if length in self.amino_dict[sequence]:
                if pos in self.amino_dict[sequence][length]:
                    return self.amino_dict[sequence][length][pos]

        dist = self.get_dist(sequence, length)
        if np.min(dist) == -1: return np.zeros((len(AMINO_ACIDS)))
        
        if sequence not in self.amino_dict: self.amino_dict[sequence] = {}
        if length not in self.amino_dict: self.amino_dict[sequence][length] = {}
        
        self.amino_dict[sequence][length][pos] = dist[pos, :]
        return dist[pos, :] 
        


def pos_entropy(probs):
    """ Computes entropy of distribution. """
    ents = np.zeros((len(probs)))
    for i in range(len(probs)):
        prob = probs[i, :]
        n_classes = np.count_nonzero(probs)
      
        if n_classes <= 1:
          ents[i] = 0
        
        # Compute entropy
        nz_prob = prob[prob > 0]
        ents[i] = np.sum(nz_prob * np.log(nz_prob))
  
    
    dist = np.exp(ents)/sum(np.exp(ents))
    return dist
