import argparse
import torch
import gym
import itertools
import numpy as np
import copy
import random
import time
import csv
from mhcflurry import Class1PresentationPredictor
from contextlib import contextmanager
import sys, os
from mhcflurry.common import configure_tensorflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AMINO_ACIDS, ALLELE_LENGTH, PEP_LENGTH, LENGTH_DIST
from stable_baselines3.common.env_util import make_vec_env
from data_utils import num2seq, seq2num
cwd = os.path.dirname(os.path.abspath(__file__))

import pdb

class PeptideEnv(gym.Env):
    def __init__(self, 
        action_space: gym.spaces,
        observation_space: gym.spaces, 
        w1: float = 0.1,
        w2: float = 0.1,
        max_len: int = 15,
        min_len: int = 8,
        max_step: int = 8,
        rate: int = 10,
        reward_type: str="game", 
        allow_imm_rew: int=None,
        mod_pos_penalty: float = -0.5,
        allow_final_rew: bool = True,
        good_sample: bool = False,
        good_sample_step: int = 50000,
        stop_criteria: float = 0.5,
        threshold: float = 0.05,
        no_mod_penalty: float=-0.3,
        anneal_nomod_step: int=0,
        anneal_nomod_rate: float=0.05,
        mod_neg_penalty: float=-0.5,
        positive_threshold: float=1.0,
        allele_name: str = "class1_pseudosequences.csv",
        discount_penalty: float=0.8,
        terminal: bool = False,
        sample_rate: float = 0.8,
        use_step: bool = False,
    ):
        super(PeptideEnv, self).__init__()
        self.possible_amino_types = np.array(AMINO_ACIDS, dtype=object)

        self.action_space = action_space
        self.observation_space = observation_space
        
        configure_tensorflow(backend='tensorflow-cpu')
        self.predictor = Class1PresentationPredictor.load()
        
        self.positive_threshold = positive_threshold
        self.allow_imm_rew = allow_imm_rew
        self.rate = rate
        self.terminal = terminal
        self.reward_type = reward_type
        self.sample_rate = sample_rate
        
        self.stop_criteria = stop_criteria
        self.use_step = use_step
        self.max_step = max_step
        self.threshold = threshold
        self.final_rew = allow_final_rew
        self.no_mod_penalty = no_mod_penalty
        self.mod_neg_penalty = mod_neg_penalty
        self.mod_pos_penalty = mod_pos_penalty
        self.discount_penalty = discount_penalty
        
        self.anneal_nomod_step = anneal_nomod_step
        self.anneal_nomod_rate = anneal_nomod_rate
                  
        # load supported alleles
        allele_path = os.path.join(os.path.dirname(cwd), 'data', allele_name)  # ALLELES
        #print(allele_path)
        # whether use good samples or not
        self.good_sample = good_sample
        self.good_sample_step = good_sample_step
        if good_sample:
            good_sample_path = os.path.join(os.path.dirname(cwd), 'data', 'good_samples.txt')
            self.good_samples = [line.split(",") for line in open(good_sample_path, 'r').readlines()]
        
        self.alleles_dict = {}
        self.alleles_seq_dict = {}
        tmp_dict = {}
        
        for line in open(allele_path, 'r').readlines()[1:]:
            elems = line.strip().split(",")
            tmp_dict[elems[0]] = elems[1]
        
        for allele in self.predictor.supported_alleles:
            tmp_allele = allele.replace("*", "")
            if tmp_allele in tmp_dict:
                self.alleles_dict[tmp_dict[tmp_allele]] = allele
                self.alleles_seq_dict[allele] = tmp_dict[tmp_allele]
                self.alleles_seq_dict[tmp_allele] = tmp_dict[tmp_allele]
                
        self.alleles = list(self.alleles_dict.keys())
        


        ## load peptides
        peptide_path = os.path.join(os.path.dirname(cwd), 'data', 'mhcflurry', 'Data_S3.csv')
        #self.alleles_with_data = []
        self.peptides = set()
        self.alleles_with_data = set()
        for line in open(peptide_path, 'r').readlines():
            elems = line.strip().split(",")
            if len(elems[1]) > 15 or len(elems[1]) < 8: continue
            if "X" in elems[1]: continue
            
            if elems[0] not in self.alleles_seq_dict: continue
            self.peptides.add((elems[0], elems[1]))
            self.alleles_with_data.add(elems[0])
             
            #tmp_allele = elems[0].replace("*", "")
            #if elems[0] in tmp_dict:
            #    self.alleles_with_data.append(tmp_dict[elems[0]])
            #elif tmp_allele in tmp_dict:
            #    self.alleles_with_data.append(tmp_dict[tmp_allele])
            
            #self.peptides.add(elems[1])
        
        self.peptides = list(self.peptides)
        
        self.len_step = 0    
        self.max_len = max_len
        self.min_len = min_len
        self.w1 = w1
        self.w2 = w2
        
        self.init_peptide = None
        self.num_step = 0
        print("finish build")

    def _get_itd_samples(self, num=1):
        """ Note: not used in recent model
            Due to that most peptides cannot receive rewards from MHCflurry predictor,
            it is necessary to find meaningful samples from datasets with neither high nor low binding affinity
            so that wrong actions can receive large negative rewards and correct actions can receive large positive rewards.
        """
        while True:
            idxs = np.random.choice(len(self.good_samples), num)
        
            redo = False
            for idx in idxs:
                if self.good_samples[idx][1] not in self.alleles_seq_dict:
                    redo = True
                    break

            if not redo: break
        
        peptides = [self.good_samples[idx][0] for idx in idxs]
        alleles  = [self.alleles_seq_dict[self.good_samples[idx][1]] for idx in idxs]
        
        return peptides, alleles
   
    def reset_with_allele(self, allele):
        self.needs_reset = False
        
        allele_seq = self.alleles_seq_dict[allele]
        peptide_idxs = np.random.choice(len(self.peptides), 1)
        peptides = [self.peptides[i] for i in peptide_idxs]

        _, _, _, present_score, _ = self._get_reward(peptides, [allele_seq])

        alleles = seq2num([allele_seq], True)
        peptides = seq2num(peptides, False)
        
        self.len_step = 0
        #self.last_score = present_score[0]
        
        if self.use_step:
            self.state = torch.cat((peptides, alleles, torch.LongTensor([[self.len_step]])), dim=1)
        else:
            self.state = torch.cat((peptides, alleles), dim=1)
        
        return self.state.squeeze(0)
        

    def reset(self, allele=None, init_peptide=None):
        if allele is None:
            idx = np.random.choice(len(self.alleles), 1)
            alleles = [self.alleles[i] for i in idx]
            self.allele = alleles[0]
        else:
            self.allele = allele
            if allele in self.alleles_seq_dict:
                alleles = [self.alleles_seq_dict[allele]]
            else:
                print(allele)
        
        self.last_score = 0
        
        ratio = random.random()
        
        present_score = None
        while True:
            if init_peptide is None:
                if ratio < self.sample_rate and (allele is None or allele in self.alleles_with_data):
                    if allele is not None:
                        peptide_idxs = [idx for idx in range(len(self.peptides)) if self.peptides[idx][0] == allele]
                        original_peptides = [self.peptides[peptide_idxs[i]][1] for i in np.random.choice(len(peptide_idxs), 1)]
                    else:
                        peptide_idxs = np.random.choice(len(self.peptides), 1)
                        original_peptides = [self.peptides[i][1] for i in peptide_idxs]


                    peptides = [None for _ in original_peptides]
                    for i, peptide in enumerate(original_peptides):
                        mutate_num = np.random.choice(np.arange(len(peptide)), np.random.randint(2, high=5, size=1)[0])
                        for idx in mutate_num:
                            if idx < len(peptide) - 1: peptides[i] = peptide[:idx] + np.random.choice(AMINO_ACIDS, 1)[0] + peptide[idx+1:]
                            else: peptides[i] = peptide[:idx] + np.random.choice(AMINO_ACIDS, 1)[0]

                    if allele is None:
                        alleles = [self.alleles_seq_dict[self.peptides[i][0]] for i in peptide_idxs]
                else:
                    peptide_len = np.random.choice(np.arange(8, 16), 1, p=LENGTH_DIST)
                    peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in peptide_len]
            else:
                peptides = [init_peptide]
                self.init_peptide = init_peptide
                break

            _, _, _, present_score, _, _ = self._get_reward(peptides, alleles)
            if present_score[0] < 0.5:
                if init_peptide is None and ratio < self.sample_rate and (allele is None or allele in self.alleles_with_data):
                    self.init_peptide = original_peptides[0]
                else:
                    self.init_peptide = None
                break
        
        
        alleles = seq2num(alleles, True)
        peptides = seq2num(peptides, False)
        
        self.len_step = 0
        self.last_score = present_score[0]
        
        if self.use_step:
            self.state = torch.cat((peptides, alleles, torch.LongTensor([[self.len_step]])), dim=1)
        else:
            self.state = torch.cat((peptides, alleles), dim=1)
        
        return self.state.squeeze(0)


    def check_terminal(self, score, actions=None):
        """ check whether the current state should be terminate or not
        """
        if self.terminal:
            if (actions is None or actions[0][0] != 1) and self.len_step < self.max_step:
                return False
            else:
                return True
        else:
            if score < self.stop_criteria and self.len_step < self.max_step:
                return False
            else:
                return True

    def mol_reward(self, score, action=None):
        """ Note: not used in recent model
            Use the reward design proposed for molecule optimization.
            
            r = \gamma^(T-t) * score
            
            The final state should be rewarded most heavily.
        """
        discount = (self.discount_penalty ** (self.max_step - self.len_step))

        reward = score * discount

        if action is not None and action[0] == 1: reward += self.no_mod_penalty

        return reward


    def game_reward(self, score, action=None):
        """ Note: not used in recent model
            if improvement is greater than threshold -> positive reward; otherwise, negative reward
        """
        if score - self.last_score[0].loc[0] >= self.threshold:
            return self.mod_pos_penalty
        else:
            return self.mod_neg_penalty


    def _get_reward(self, peptides, alleles, actions=None):
        """ calculate the reward
        """
        allele_names = [self.alleles_dict[allele] for allele in alleles]
        predictions = self.predictor.predict(peptides=peptides, alleles=allele_names, verbose=0)
        
        affinity = 1 - np.log(predictions['affinity']) / np.log(50000)
        process_score = predictions['processing_score']
        present_score = predictions['presentation_score']
        
        rewards, diffs = [],[]
        terminals = []
        self.num_step += 1
        
        for i in range(len(peptides)):
            terminal = self.check_terminal(present_score[i], actions=actions)
                     
            reward = 0
            
            if actions is None:
                reward = 0
            elif not (self.final_rew and terminal) and (self.allow_imm_rew is None or self.num_step < self.allow_imm_rew): 
                if self.reward_type == "game": reward = self.game_reward(present_score[i], actions[i])
                elif self.reward_type == "molecule": reward = self.mol_reward(present_score[i], actions[i])    
            elif terminal:
                reward = present_score[i] * self.rate
            
            terminals.append(terminal)
            rewards.append(reward)
            
            diff = present_score[i] - self.last_score
            self.last_score = present_score[i]
        
        
        return rewards, affinity, process_score, present_score, terminals, diff
    
    def use_mhcflurry_predict(self, peptides, alleles):
        """ calculate the scores using mhcflurry
        """
        allele_names = {i:[self.alleles_dict[allele]] for i, allele in enumerate(alleles)}
        
        sample_names = [i for i in range(len(alleles))]
        
        scores = self.predictor.predict(peptides=peptides, alleles=allele_names, sample_names=sample_names, verbose=0)
        return scores['presentation_score']


    #def get_difference(self, peptides, actions, alleles):
    #    """
    #    """
    #    new_peptides = self._edit_sequence(peptides, actions)
    #
    #    old_scores = self.use_mhcflurry_predict(peptides, alleles)
    #    new_scores = self.use_mhcflurry_predict(new_peptides, alleles)
    #
    #    return new_scores - old_scores


    def _edit_sequence(self, peptides, actions):
        new_peptides = []
        for i in range(len(peptides)):
            peptide = peptides[i]
            action = actions[i, :]
            
            position = action[0]
            new_peptide = ""
            
            new_peptide = peptide[:position] + AMINO_ACIDS[action[1]-1]
                
            if position < len(peptide) - 1:
                new_peptide += peptide[position+1:]
                
            new_peptides.append(new_peptide)
        
        return new_peptides
    
    def step(self, actions: torch.Tensor):
        """ State Transition
        """
        peptides = num2seq(self.state[:, :15], False)
        alleles = num2seq(self.state[:, 15:49], True)
        
        ### take action
        if len(actions.shape) == 1: actions = np.expand_dims(actions, axis=0)
        new_peptides = self._edit_sequence(peptides, actions)
         
        self.len_step += 1
        term_rewards, affinity, process_score, present_score, terminals, diff = self._get_reward(new_peptides, alleles, actions=actions)
        
        info = {}
        info['terminal'] = ",".join([str(terminals[i]) for i in range(len(terminals))])
        info['action'] = ",".join([str(actions[0][i]) for i in range(2)])
        info['old_peptide'] = ",".join(peptides)
        info['new_peptides'] = ",".join(new_peptides)
        info['allele'] = ",".join([self.allele])
        info['rewards'] = ",".join(["%.4f" % (reward) for reward in term_rewards])
        info['process_score'] = ",".join(["%.4f" % (score) for score in process_score])
        info['present_score'] = ",".join(["%.4f" % (score) for score in present_score])
        info['affinity'] = ",".join(["%.4f" % (score) for score in affinity])
        info['difference'] = diff
         
        if self.use_step:
            self.state[:, -1] = self.len_step
        
        self.state = torch.cat((seq2num(new_peptides, False), self.state[:, 15:]), dim=1)
        
        return self.state.squeeze(0), term_rewards[0], terminals[0], info

if __name__ == '__main__':
    import sys, os
    from ppo import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from policy import PolicyNet
    from seq_embed import SeqEmbed
    import pickle
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="path to save results")
    parser.add_argument('--allele_name', type=str, default="class1_pseudosequences.csv", help="path of alleles") 
    
    # reward design
    parser.add_argument('--reward_type', type=str, default="game", help="select reward for game or molecule")
    parser.add_argument('--terminal', action="store_true", help="whether using the no-modification action as termination")
    parser.add_argument('--discount_penalty', type=float, default=0.8, help="used for molecule modification-based reward design")
    
    parser.add_argument('--mod_pos_penalty', type=float, default=0, help="penalty for each step")
    parser.add_argument('--no_mod_penalty', type=float, default=-0.5, help="penalty for no modification")
    parser.add_argument('--mod_neg_penalty', type=float, default=0, help="penalty for negative modification")
    
    parser.add_argument('--allow_imm_rew', type=int, default=None, help="whether use immediate reward or not: (None represent using imm reward; 0 represent not using imm reward")
    parser.add_argument('--allow_final_rew', action="store_false", help="whether use final reward or not")

    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--rate', type=float, default=10, help="weight of final reward")
    
    parser.add_argument('--anneal_nomod_step', type=int, default=10000)
    parser.add_argument('--anneal_nomod_rate', type=float, default=0.05)
    
    # parameters for policy network
    parser.add_argument('--allele_hidden_dim', type=int, default=320, help="dimension of hidden units in allele model")
    parser.add_argument('--allele_latent_dim', type=int, default=80, help="dimension of latent allele embeddings")
    parser.add_argument('--allele_kmer', type=int, default=3, help="need to be specified when using CNN to encode alleles")
    parser.add_argument('--embed_allele', type=str, default="FC", help="model architecture used to encode alleles")
    
    parser.add_argument('--peptide_latent_dim', type=int, default=40, help="dimension of latent amino embeddings in peptides")
    parser.add_argument('--shared_dim', type=int, default=40, help="dimension of shared layer in policy network and value network")
    parser.add_argument('--latent_dim', type=int, default=20, help="dimension of latent layer in two networks")

    # ppo algorithms
    parser.add_argument('--good_coef', type=float, default=0.005, help="coef for good loss")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    parser.add_argument('--steps', type=int, default=10000000, help="total time steps")
    parser.add_argument('--n_steps', type=int, default=512, help="number of roll out steps. Number of time steps in each batch")
    
    parser.add_argument('--threshold', type=float, default=0.05, help="threshold of positive reward")
    parser.add_argument('--positive_threshold', type=float, default=1.0, help="positive threshold")
    parser.add_argument('--ent_coef', type=float, default=0.1, help="encourage exploration")
    parser.add_argument('--pretrain_iter', type=int, default=3000, help="the iteration for pretraining")
    
    parser.add_argument('--clip', type=float, default=0.2, help="parameter for ppo algorithm")
    parser.add_argument('--kl_target', type=float, default=0.01, help="parameter for ppo algorithm")
    
    # environment
    parser.add_argument('--num_envs', type=int, default=30)
    parser.add_argument('--max_len', type=int, default=15, help="maximum length of peptides")
    parser.add_argument('--use_step', action="store_true", help="whether to use step information as a feature in action prediction or not") 
    parser.add_argument('--stop_criteria', type=float, default=1.0, help="stop_criteria")
    parser.add_argument('--max_step', type=int, default=8, help="maximum number of steps for each peptide; the maximum number of mutations")
    parser.add_argument('--good_sample_step', type=int, default=1, help="good sample step")
    parser.add_argument('--sample_rate', type=float, default=0.8, help='the rate of initial peptides sampled from IEDB dataset')
    args = parser.parse_args()
    
    name = ""
    for arg in vars(args):
        if arg == "path": continue
        attr = getattr(args, arg)
        if isinstance(attr, str):
            name += attr+"_"
        else:
            name += str(attr)+"_"
    
    path = args.path + name[:-1]
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    #sys.stdout = open(args.path + name[:-1] + "/log", 'w+')
    
    action_space = gym.spaces.multi_discrete.MultiDiscrete([15, 20])
    if args.use_step:
        observation_space = gym.spaces.MultiDiscrete([20] * 50)
    else:
        observation_space = gym.spaces.MultiDiscrete([20] * 49)
    
    m_env_kwargs = {"action_space":action_space, "observation_space":observation_space, \
                    "good_sample_step":args.good_sample_step, \
                    "no_mod_penalty":args.no_mod_penalty, "mod_pos_penalty":args.mod_pos_penalty, \
                    "allow_final_rew":args.allow_final_rew, "allow_imm_rew":args.allow_imm_rew, \
                    "allele_name":args.allele_name, "mod_neg_penalty": args.mod_neg_penalty, \
                    "threshold":args.threshold, "rate":args.rate, "reward_type":args.reward_type, \
                    "discount_penalty":args.discount_penalty, "anneal_nomod_step":args.anneal_nomod_step, \
                    "anneal_nomod_rate":args.anneal_nomod_rate, "terminal":args.terminal, "sample_rate":args.sample_rate, \
                    "stop_criteria":args.stop_criteria, "max_step":args.max_step, "max_len":args.max_len, \
                    "positive_threshold":args.positive_threshold}
    
    ftype = {"deep":True, "blosum":True, "onehot": True}
    config = {"ftype":ftype, "embed_dim": 60,
              "allele_hidden_dim": args.allele_hidden_dim, "allele_latent_dim": args.allele_latent_dim, \
              "allele_kmer": args.allele_kmer, "embed_allele": args.embed_allele, \
              "peptide_latent_dim": args.peptide_latent_dim, "use_step": args.use_step}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    seq_features = SeqEmbed(config)
    
    policy_kwargs = dict(features_extractor=seq_features, \
                        net_arch = [args.shared_dim, dict(vf=[args.latent_dim], pi=[args.latent_dim])], \
                        use_step = args.use_step)
    
    checkpoint_callback = CheckpointCallback(save_freq=200, save_path=path+'/', name_prefix='rl_model')
    
    m_env = make_vec_env(PeptideEnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs, vec_env_cls=SubprocVecEnv)
    
    good_buffer_config = dict(max_len = 20000, ncpu=args.num_envs)
    model = PPO(PolicyNet, m_env, verbose=1, n_steps=args.n_steps, ent_coef=args.ent_coef, good_coef=args.good_coef, gamma=args.gamma, clip_range=args.clip, target_kl=args.kl_target, good_buffer_config=good_buffer_config, policy_kwargs=policy_kwargs)
    
    pretrain_env = PeptideEnv(**m_env_kwargs)
    model.learn(pretrain_iter=args.pretrain_iter, pretrain_env=pretrain_env, total_timesteps=args.steps, callback=checkpoint_callback)
    
    model.save(path+"/ppo_peptide")
    
