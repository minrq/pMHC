import os
import torch
#from stable_baselines3.common.utils import get_device

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
cwd = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.dirname(cwd)

ALLELE_PATH = PATH + "/data/class1_pseudosequences.csv"
ALLELE_LENGTH = 34
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
PEP_LENGTH = 15
BLOSUM = PATH + "/data/blosum62.txt"

LEARNED_DIM = 20

DICT_PATH = PATH + "/data/dict3.pkl"

LENGTH_DIST = [0.15, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05]
