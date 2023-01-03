import time
import pdb
import torch
import argparse
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.env_util import make_vec_env

from ppo import PPO
from peptide_env import PeptideEnv
from data_utils import num2seq, seq2num
import torch

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
import config

#config.device = "cuda" if torch.cuda.is_available() else "cpu"

def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data[0])

                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    # observation = env.reset(allele=data[1])

                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                if data[0]:
                    observation = env.reset(allele=data[1], init_peptide=data[3])
                else:
                    observation = data[2]
                
                remote.send(observation)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class MySubprocVecEnv(VecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def step(self, actions: np.ndarray, alleles: list, peptides: list) -> VecEnvStepReturn:
        """
        Step the environments with the given action
        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions, alleles, peptides)
        return self.step_wait()       

    def step_async(self, actions: np.ndarray, alleles: list, peptides: list) -> None:
        for remote, action, allele, peptide in zip(self.remotes, actions, alleles, peptides):
            remote.send(("step", (action, allele, peptide)))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        for info in infos:
            for key, value in info.items():
                if key == "episode" or key == "terminal_observation": continue
                print("%s: %s; " % (key, value), end='')
            print("\n") 
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self, dones, alleles, obs, peptides) -> VecEnvObs:
        for remote, done, allele, ob, peptide in zip(self.remotes, dones, alleles, obs, peptides):
            remote.send(("reset", (done, allele, ob, peptide)))
        
        obs = [remote.recv() for remote in self.remotes]
        sys.stdout.flush()
        return _flatten_obs(obs, self.observation_space)
    


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alleles", type=str)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--path", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument('--peptide', type=str, default=None)
    
    parser.add_argument('--stop_criteria', type=float, default=0.75, help="stop_criteria")
    parser.add_argument('--num_envs', type=int, default=3, help="number of environments")
    parser.add_argument('--max_step', type=int, default=8, help="maximum number of steps")
    parser.add_argument('--sample_rate', type=float, default=0.5, help='the rate of sampling from IEDB dataset')
    parser.add_argument('--use_step', action="store_true")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--hour', type=int, default=5)
    parser.add_argument('--max_size', type=int, default=500)
     
    args = parser.parse_args()
    t1 = time.time()
    
    alleles = [allele.strip() for allele in open(args.alleles, 'r').readlines()]
    
    action_space = gym.spaces.multi_discrete.MultiDiscrete([15, 20])
    
    if args.use_step:
        observation_space = gym.spaces.MultiDiscrete([20] * 50)
    else:
        observation_space = gym.spaces.MultiDiscrete([20] * 49)
    
    if args.peptide:
        peptides = [peptide.strip() for peptide in open(args.peptide, 'r').readlines()]
    else:
        peptides = None
    
    m_env_kwargs = {"action_space":action_space, "observation_space":observation_space, \
                "sample_rate":args.sample_rate, "stop_criteria":args.stop_criteria, 
                "max_step":args.max_step}
    
    m_env = make_vec_env(PeptideEnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs, vec_env_cls=MySubprocVecEnv)
   
    model = PPO.load(args.path,env=m_env)
    
    results = {allele: [] for allele in alleles}
    
    if args.peptide:
        rollout_peptides = [peptide for peptide in peptides for _ in alleles]
        rollout_alleles = [allele for allele in alleles] * len(peptides)
    else:
        rollout_alleles = [allele for allele in alleles] * args.rollout
        rollout_peptides = [None] * len(rollout_alleles)
    #rollout_alleles = [allele for tmp in rollout_alleles for allele in tmp]
    
    batch_alleles = rollout_alleles[:args.num_envs]
    batch_peptides = rollout_peptides[:args.num_envs]
    batch_idxs = np.arange(args.num_envs)
    
    obs = m_env.reset([True] * len(batch_alleles), batch_alleles, [None] * len(batch_alleles), batch_peptides)
    rollout = 0

    st_time = time.time()
    num = len(alleles)
    removed_alleles = []
    while rollout < len(rollout_alleles):
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, config.device)
            actions, values, log_probs = model.policy.forward(obs_tensor)
        
        actions = actions.cpu().numpy()
        
        new_obs, rewards, dones, infos = m_env.step(actions, batch_alleles, batch_peptides)
        
        for idx, done in enumerate(dones):
            if done:
                allele = rollout_alleles[batch_idxs[idx]]
                #if allele != infos[idx]['allele']: pdb.set_trace()
                if max(batch_idxs) < len(rollout_alleles) - 1:
                    batch_idxs[idx] = max(batch_idxs) + 1
                    batch_alleles[idx] = rollout_alleles[batch_idxs[idx]]
                    batch_peptides[idx] = rollout_peptides[batch_idxs[idx]]
                
                rollout += 1

                if rollout < len(rollout_alleles):
                    #if float(infos[idx]['present_score']) >= args.stop_criteria:
                    results[allele].append((infos[idx]['new_peptides'], infos[idx]['present_score'], infos[idx]['process_score'], infos[idx]['affinity']))
                else:
                    break
       
        #pdb.set_trace()
        obs = m_env.reset(dones, batch_alleles, new_obs, batch_peptides) 
        ck_time = time.time()
        for allele in results:
            if allele not in removed_alleles and len(results[allele]) >= args.max_size:
                num -= 1
                removed_alleles.append(allele)
                print("time cost for allele %s: %.4f" % (allele, ck_time - st_time))

        if num == 0: break
        if ck_time - st_time >= args.hour * 3600: break
        
        obs = new_obs
        
    m_env.close()
    output = open(args.out, 'w')
            
    for allele in alleles:
        for result in results[allele]:
            output.write("%s %s %.4f %.4f %.4f\n" % (allele, result[0], \
                      float(result[1]), float(result[2]), float(result[3])))

    output.close()
    print("time cost: %.4f" % (time.time()-t1))
