import copy
from collections import deque, defaultdict
import numpy as np
import heapq
import pdb
from config import AMINO_ACIDS
import torch
from multiprocessing import Pool

class GoodBuffer:
    def __init__(self, args={}):
        if args is None: args = {'max_len':1000}
        self.config = args
        if "good_sample" in args:
            self.initialize(args['good_sample'])
         
        if "max_len" in args:
            self._states = []
            self._actions = []
        
        self._popular = {}
        self._counter = defaultdict(int)
        self._indices = {}
        
    def length(self):
        return len(self._states)
 
    def store(self, states, actions):
        alleles = [state[:34] for state in states]
        
        count_aminos = self.get_count_adds(alleles, actions)
        
        # get the number of actions that need to be removed
        remove_num = 0
        for allele in count_aminos:
            for action in count_aminos[allele]:
                if action in self._popular and self._popular[action] >= 0.02:
                    remove_num += count_aminos[allele][action]
        
        num = (max(len(self._states) + len(alleles) - self.config['max_len'] - remove_num, 0))
        remove_num += num
        
        print("need to remove %d actions; buffer size: %d" % (remove_num, len(self._states)))
        idxs = []
        sorted_list = sorted(self._counter, key=lambda x: self._counter[x], reverse=True)
        
        if remove_num > 0:
            for i, action in enumerate(sorted_list):
                num = self._counter[action]
                print("remove action %d %d" % (action[0], action[1]))
                
                if i < len(sorted_list) - 1:
                    diff = num - self._counter[sorted_list[i+1]]
                else:
                    diff = num
                
                if remove_num > diff * (i+1):
                    for j, remove_action in enumerate(sorted_list[:i+1]):
                        idxs.extend(self._indices[remove_action][:diff])
                        self._indices[remove_action] = self._indices[remove_action][diff:]
                        
                    remove_num -= diff * (i+1)
                else:
                    for j, remove_action in enumerate(sorted_list[:i+1]):
                        if remove_num > diff:
                            idxs.extend(self._indices[remove_action][:diff])
                            self._indices[remove_action] = self._indices[remove_action][diff:]
                            remove_num -= diff
                        else:
                            idxs.extend(self._indices[remove_action][:remove_num])
                            self._indices[remove_action] = self._indices[remove_action][remove_num:]
                            remove_num = 0
                            break
                    break
                
        
        if len(idxs) < len(alleles):
            num = len(alleles) - len(idxs)
            extend_idxs = [i for i in range(len(self._states), num + len(self._states))]
        
        count_removes = self.get_count_removes(idxs)
        
        for i, idx in enumerate(idxs):
            self._states[idx] = states[i]
            self._actions[idx] = actions[i]
            action = tuple(actions[i])
            if action not in self._indices: self._indices[action] = []
            self._indices[action].append(idx) 
        
        for i, action in enumerate(actions[len(idxs):]):
            if tuple(action) not in self._indices: self._indices[tuple(action)] = []
            self._indices[tuple(action)].append(len(self._states) + i)
        
        self._states.extend(states[len(idxs):])
        self._actions.extend(actions[len(idxs):])
        
        self.update_popular(count_aminos, count_removes)

    def sample(self, batch_size):
        idxs = np.random.choice(self._unpopular_idxs, batch_size)

        states = torch.Tensor([self._states[i] for i in idxs])
        actions = torch.Tensor([self._actions[i] for i in idxs])
        
        return states, actions
        
    def get_count_adds(self, alleles, actions):
        count_aminos = {}
        for i, (allele, action) in enumerate(zip(alleles, actions)):
            allele = tuple(allele)
            if allele not in count_aminos:
                count_aminos[allele] = {}
            
            if tuple(action) not in count_aminos[allele]:
                count_aminos[allele][tuple(action)] = 0
            
            count_aminos[allele][tuple(action)] += 1
        
        return count_aminos
        
    def get_count_removes(self, idxs):
        count_removes = {}
        for idx in idxs:
            allele = tuple(self._states[idx][:34])
            action = tuple(self._actions[idx])
            
            if allele not in count_removes: count_removes[allele] = {}
            if action not in count_removes[allele]: count_removes[allele][action] = 0
            count_removes[allele][action] += 1
        return count_removes

    def update_popular(self, count_aminos, count_removes):
        alleles = list(count_aminos.keys() | count_removes.keys())

        for allele in alleles:
            if allele in count_aminos:
                for amino in count_aminos[allele]:
                    self._counter[amino] += count_aminos[allele][amino]
                    

            if allele in count_removes:
                for amino in count_removes[allele]:
                    self._counter[amino] -= count_removes[allele][amino]
                        
        total_examples = sum([self._counter[action] for action in self._counter])
        
        if total_examples <= 500:
            for action in self._counter:
                self._popular[action] = 0
        else:
            for action in self._counter:
                self._popular[action] = self._counter[action] / total_examples
                #print("popularity of action %d, %d: %.4f" % (action[0], action[1], self._popular[action]))
        
        self._unpopular_idxs = [idx for action in self._popular if self._popular[action] < 0.02 for idx in self._indices[action]]

    def update_priority(self):
        self._priority = np.zeros((len(self._states)))
        for i, (state,action) in enumerate(zip(self._states, self._actions)):
            self._priority[i] = self._popular[action[1]-1]

        self._priority = list(self._priority)
