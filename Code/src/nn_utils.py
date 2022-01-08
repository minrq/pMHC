import random
import pickle
from itertools import zip_longest
from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pdb
from gym import spaces
import torch
from torch import nn
from data_utils import num2seq

from config import PATH, DICT_PATH
from prior_dist import PriorDist
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution)

from torch.distributions.categorical import Categorical

class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
        use_step: bool = False,
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, amino_net, policy_net, value_net = [], [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        
        self.use_step = use_step

        if self.use_step:
            last_layer_dim_shared = feature_dim + 1
        else:
            last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                amino_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                amino_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                amino_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        #pdb.set_trace()
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.policy_amino_net = nn.Sequential(*amino_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: Tuple) -> Tuple[Tuple, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        if self.use_step:
            pep_features, allele_embeds, steps = features
        else:
            pep_features, allele_embeds = features
        
        # Policy features
        amino_features, pep_features = pep_features
        amino_embeds, pep_lengths = amino_features
         
        # Amino Acid Features
        selected_allele_embeds = allele_embeds.repeat(amino_embeds.shape[0], 1, 1)
        if not self.use_step:
            amino_features = torch.cat((amino_embeds, selected_allele_embeds), dim=2)
        else:
            _steps = steps.repeat(amino_embeds.shape[0], 1).unsqueeze(2)
            amino_features = torch.cat((amino_embeds, selected_allele_embeds, _steps), dim=2)
        
        latent_policy_aminos = self.policy_amino_net(amino_features)

        # Peptide Features
        pep_features = torch.flatten(pep_features.transpose(1, 0), start_dim=1)
        if self.use_step:
            _steps = steps.unsqueeze(1)
            pep_features = torch.cat((pep_features, allele_embeds, _steps), dim=1)
        else:
            pep_features = torch.cat((pep_features, allele_embeds), dim=1)
        
        latent_pep_features = self.shared_net(pep_features)
        latent_policy_peps = self.policy_net(latent_pep_features)
        latent_vf = self.value_net(latent_pep_features)
        
        latent_pi = (latent_policy_aminos, latent_policy_peps)
        return latent_pi, latent_vf


class PeptideActionNet(nn.Module):
    def __init__(self, 
        latent_dim: int,
        action_space: spaces,
        device: Union[torch.device, str] = "auto", 
    ):
        super(PeptideActionNet, self).__init__()
        self.action_space = action_space
        self.latent_dim = latent_dim
        self.device = get_device(device)
        
        #self.stop_action_net = nn.Linear(latent_dim, 2).to(self.device)
        #self.stop_dist = self._build_dist(2)
        
        self.pos_action_net = nn.Linear(latent_dim, 1).to(self.device)
        self.pos_dist = self._build_dist(15)
        
        self.amino_action_net = nn.Linear(latent_dim, 20).to(self.device)
        self.amino_dist = self._build_dist(20)
        
        self.prior_dist = PriorDist(device)
        
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, latent_pi: Tuple, peptides: torch.Tensor, alleles: torch.Tensor, lengths: torch.Tensor, pretrain=False)->torch.Tensor:
        latent_amino, latent_pep = latent_pi
        batch_size = len(peptides)
        #lengths = lengths.to(self.device)
        
        # get the probability distribution of position prediction
        pack_latent_amino = nn.utils.rnn.pack_padded_sequence(latent_amino, lengths, enforce_sorted=False)
        flat_pos_pd = self.pos_action_net(pack_latent_amino.data)
        pack_pos_pd = nn.utils.rnn.PackedSequence(
                          flat_pos_pd, 
                          pack_latent_amino.batch_sizes,
                          pack_latent_amino.sorted_indices,
                          pack_latent_amino.unsorted_indices)
               
        pos_pd, _ = nn.utils.rnn.pad_packed_sequence(pack_pos_pd, padding_value=-100000)
        
        pos_pd = pos_pd.squeeze(2).transpose(1, 0)
        pos_dist = self.pos_dist.proba_distribution(pos_pd)
        
        if pretrain:
            pos_ac = self.sample_with_prior(pos_pd.shape[1], alleles, peptides, lengths, if_pos=True)
        else:
            pos_ac = pos_dist.sample()
            pos_logpd = pos_dist.log_prob(pos_ac)
        
        # get the amino scores of peptide sequences
        select_amino_idx = pos_ac.unsqueeze(1).repeat(1, self.latent_dim).unsqueeze(0)
        amino_features = torch.gather(latent_amino, 0, select_amino_idx).squeeze(0)
        amino_pd = self.amino_action_net(amino_features)
        
        mask_amino = torch.gather(peptides, 1, pos_ac.unsqueeze(1)).squeeze(1) - 1
        #zero_amino_mask = torch.stack((mask_amino_idx, mask_amino), dim=1)
        amino_pd[torch.arange(len(mask_amino)).to(self.device), mask_amino] = -100000
        
        amino_dist = self.amino_dist.proba_distribution(amino_pd)

        if pretrain:
            amino_ac = self.sample_with_prior(amino_pd.shape[1], alleles, peptides, lengths, if_pos=False, pos=pos_ac, mask_amino=mask_amino)
        else:
            amino_ac = amino_dist.sample()
            amino_logpd = amino_dist.log_prob(amino_ac)
            amino_ac += 1
        
        #action = nn.utils.rnn.pad_sequence([pos_ac, amino_ac])
        #logpd = nn.utils.rnn. pad_sequence([pos_logpd, amino_logpd])
        
        action = nn.utils.rnn.pad_sequence([pos_ac, amino_ac])
        
        if pretrain:
            return action, pos_pd, amino_pd
        
        logpd = nn.utils.rnn.pad_sequence([pos_logpd, amino_logpd])
        logpd = torch.sum(logpd, dim=1)
        return action, logpd
    
    def evaluate_actions(self, latent_pi: Tuple, actions: torch.Tensor, peptides: torch.Tensor, lengths: torch.Tensor):
        latent_amino, latent_pep = latent_pi
        batch_size = len(peptides)
        #lengths = lengths.to(self.device)
        
        packed_latent_amino = nn.utils.rnn.pack_padded_sequence(latent_amino, lengths, enforce_sorted=False)
        flat_pos_pd = self.pos_action_net(packed_latent_amino.data)
        
        pack_pos_pd = nn.utils.rnn.PackedSequence(flat_pos_pd, 
                          packed_latent_amino.batch_sizes, 
                          packed_latent_amino.sorted_indices, 
                          packed_latent_amino.unsorted_indices)
        
        pos_pd, _ = nn.utils.rnn.pad_packed_sequence(pack_pos_pd, padding_value=-10000)
        
        pos_pd = pos_pd.squeeze(2).transpose(1, 0)
        pos_dist = self.pos_dist.proba_distribution(pos_pd)
        
        pos_ac = actions[:, 0]
        try:
            pos_logpd = pos_dist.log_prob(pos_ac)
            pos_entro = pos_dist.entropy()
        except:
            pdb.set_trace()
        
        amino_idx = pos_ac.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.latent_dim).long()
        
        amino_features = torch.gather(latent_amino, 0, amino_idx).squeeze(0)
        amino_pd = self.amino_action_net(amino_features)
        
        mask_amino = torch.gather(peptides, 1, pos_ac.unsqueeze(1).long()).long().squeeze(1) - 1
        #zero_amino_mask = torch.stack((mask_amino_idx, mask_amino), dim=1)
        amino_pd[torch.arange(len(mask_amino)).to(self.device), mask_amino] = -10000
        
        amino_dist = self.amino_dist.proba_distribution(amino_pd)
        #amino_action_id = actions[:, 2].nonzero().squeeze(1)
        #amino_action = actions[amino_action_id, 2] - 1
        amino_action = actions[:, 1] - 1
       
        amino_logpd = amino_dist.log_prob(amino_action)
        amino_entro = amino_dist.entropy()
        
        logpd = torch.stack([pos_logpd, amino_logpd], axis=1)
        entro = torch.stack([pos_entro, amino_entro], axis=1).sum(dim=1)
        
        logpd = torch.sum(logpd, dim=1)
        
        return logpd, entro, (pos_pd, amino_pd)
        
        
    def _build_dist(self, dim: int)->Distribution:
        """
        """
        return CategoricalDistribution(dim)

    def sample_with_prior(self, shapey, latent_allele, peptides, lengths, if_pos=True, pos=None, mask_amino=None):
        """ 
        """
        
        alleles = num2seq(latent_allele, True)
        
        new_dist = np.ones((lengths.shape[0], shapey)) * -1000
        
        for i in range(len(lengths)):
            if if_pos:
                new_dist[i, :lengths[i].item()] = self.prior_dist.get_pos(alleles[i], lengths[i].item(), peptides[i])
            elif not if_pos:
                new_dist[i, :] = self.prior_dist.get_amino(alleles[i], lengths[i].item(), pos[i].item())
                new_dist[i, peptides[i, pos[i]]-1] = 0
                new_dist[i, :] = new_dist[i, :] / sum(new_dist[i, :])
       
        #pdb.set_trace()
        if np.isnan(np.sum(new_dist)): new_dist = np.nan_to_num(new_dist)
        
        #if if_pos: new_dist = np.exp(new_dist) / np.repeat(np.expand_dims(np.sum(np.exp(new_dist), axis=1), axis=1), new_dist.shape[1], axis=1)
        
        if if_pos: ac = np.argmax(new_dist, axis=1)
        else:
            try:
                ac = (new_dist.cumsum(1) > np.random.rand(new_dist.shape[0])[:,None]).argmax(1)
            except:
                ac = np.argmax(new_dist, axis=1)
        #pdb.set_trace() 
        ac = torch.LongTensor(ac).to(self.device)
        return ac
