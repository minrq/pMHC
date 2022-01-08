import torch
import gym
import itertools
import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt
import csv
from mhcflurry import Class1PresentationPredictor
from contextlib import contextmanager
import sys, os
from config import AMINO_ACIDS, ALLELE_LENGTH, PEP_LENGTH
import pdb

class PeptideEnv(gym.Env):
    def __init__(self, 
        observation_space: gym.spaces, 
        action_space: gym.spaces,
        w1: float = 0.1,
        w2: float = 0.1,
        max_len: int = 15,
        min_len: int = 8,
        batch_size: int = 32,
    ):
        super(PeptideEnv, self).__init__()
        self.possible_amino_types = np.array(AMINO_ACIDS, dtype=object)
        self.batch_size = batch_size

        self.action_sapce = action_space
        self.observation_space = observation_space
                
        ## load expert data
        cwd = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(cwd), 'data', 'class1_pseudosequences.csv')  # ZINC
        self.alleles = [line.strip().split(",")[1] for line in open(path, 'r').readlines()][1:]
        self.predictor = Class1PresentationPredictor.load()

        self.a2n_func = {}
        self.n2a_func = {}
        for i, amino in enumerate(AMINO_ACIDS):
            self.a2n_func[amino] = i+1
            self.n2a_func[i+1] = amino
        
        self.max_len = max_len
        self.min_len = min_len
        self.w1 = w1
        self.w2 = w2


    def _seq2num(self, sequences, allele):
        """ Convert sequences to interger values 
        example: "ACT" -> [1, 2, 3]
        """
        max_len = ALLELE_LENGTH if allele else PEP_LENGTH
        arrays = np.zeros((len(sequences), max_len))
        lengths = np.zeros((len(sequences)))

        for i, seq in enumerate(sequences):
            lengths[i] = len(seq)

            for j, amino in enumerate(seq):
                arrays[i,j] = self.a2n_func[amino]
        
        arrays = torch.LongTensor(arrays)
        return arrays

    def _num2seq(self, sequences, allele):
        new_sequences = []
        if allele:
            for i in range(sequences.shape[0]):
                seq = "".join([self.n2a[idx] for idx in sequences[i,:].to_numpy()])
                new_sequences.append(seq)

        else:
            for i in range(sequences.shape[0]):
                seq = "".join([self.n2a[idx] for idx in sequences[i,:].to_numpy() if idx != 0])
                new_sequences.append(seq)

        return new_sequences
        
    def reset(self):
        idx = np.random.choice(len(self.alleles), self.batch_size)
        alleles = [self.alleles[i] for i in idx]
        
        alleles = self._seq2num(alleles, True)
        peptide_len = np.random.choice(np.arange(8, 16), self.batch_size)
        peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in peptide_len]
        
        peptides = self._seq2num(peptides, False)
        states = torch.cat((peptides, alleles), dim=1)
        return states

    def _get_reward(self, peptides, alleles, stop=False):
        predictions = self.predictor.predict(peptides=peptides, alleles=peptides, verbose=0)

        affinity = predictions['affinity']
        process_score = predictions['processing_score']
        present_score = predictions['presentation_score']
        
        if not stop:
            rewards = 0.3 * affinity
        else:
            rewards = affinity + process_score * self.w1 + present_score * self.w2
        
        return rewards
    
    def _edit_sequence(self, peptides, actions):
        new_sequences = []
        terminals = np.zeros((len(peptides)))
        
        for i in range(actions.shape[0]):
            action = actions[i,:]
            if action[0] == 1:
                 new_sequences.append(peptide)
                 terminals[i] = 1
                 continue
            
            position = action[1]
            
            # deletion
            if action[2] == 0:
                if len(peptide) == self.min_len:
                    new_sequences.append(peptide)
                    terminals[i] = 1
                    continue
                else:
                    new_sequences.append(peptide[:position] + peptide[position+1:])
            # insertion
            elif action[2] == 1:
                if len(peptide) == self.max_len:
                    new_sequences.append(peptide)
                    terminals[i] = 1
                    continue
                else:
                    new_sequences.append(peptide[:position] + AMINO_ACIDS[action[3]] + peptide[position:])
            # replacement
            elif action[2] == 2:
                new_peptide = peptide
                new_peptide[position] = AMINO_ACIDS[action[3]]
                new_sequences.append(new_peptide)
            else:
                raise ValueError("wrong action")
            
            terminals[i] = 0
        return new_sequences, terminals
        
    def step(self, actions: torch.Tensor):
        peptides = 
        
        ### take action
        new_peptides, terminal = self._edit_sequence(peptides, actions)
        
        term_rewards = self._get_reward(new_peptides, alleles, stop=terminal)

        return new_peptides, term_rewards, terminal, {}

if __name__ == '__main__':
    # env = GraphEnv()
    # env.init(has_scaffold=True)

    ## debug
    from ppo import PPO
    from policy import PolicyNet
    from seq_embed import SeqEmbed

    action_space = gym.spaces.MultiDiscrete([2, 15, 3, 20])
    observation_space = gym.spaces.MultiDiscrete([49, 20])
    
    m_env = PeptideEnv(10, action_space, observation_space)

    ftype = {"deep":True, "blosum":True, "onehot": True}
    config = {"ftype":ftype, "embed_dim":60, \
              "hidden_dim":10, "latent_dim":20, \
              "kmer":3, "embed_allele":'CNN'}
    
    seq_features = SeqEmbed(config)
    policy_kwargs = dict(features_extractor=seq_features, net_arch = [10, dict(vf=[5], pi=[5])])
    model = PPO(PolicyNet, m_env, verbose=1, policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=10000)
