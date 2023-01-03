## Binding Peptide Generation for MHC Class I Proteins with Deep Reinforcement Learning

This is the implementation of our PepPPO model for the paper below,

> Ziqi Chen, Baoyi Zhang, Hongyu Guo, Prashant Emani, Trevor Clancy, Chongming Jiang, Mark Gerstein, Xia Ning, Chao Cheng, and Martin Renqiang Min.
> Binding Peptide Generation for MHC Class I Proteins with Deep Reinforcement Learning.
> Accepted to Bioinformatics. December 2022.

Our DRL framework is based on [stable-baseline3](https://github.com/DLR-RM/stable-baselines3). 



## Requirements

* python==3.6.13
* torch==1.9.1
* tensorflow==2.2.0
* gym==0.18.0

* Mhcflurry==2.0.1

  

## Overview

<code>code</code>  source code

<code>baselines</code> code for baselines including MCTS, VAE with backpropagation and VAE with bayesian optimization.

<code>data</code> data used for training

<code>stable_baselines3</code>  code from the package "stable_baseline3". Our implementation of RL model is based on this package.



## Training

To train a PepPPO model, run

```
python ./code/peptide_env.py --path <model path> --allow_imm_rew 0 --gamma 0.9
```



## Testing

To test a trained PepPPO model with specific alleles, run

```
python ./code/test_RL.py --sample_rate 0.5 --alleles ./data/test_alleles.txt --out <test result file> --rollout 1000 --path ./model/ppo_peptide.zip
```

<code>sample_rate</code> : the percentage of initial peptides that are sampled from the dataset. For example, "sample_rate = 0.0" represents that all the initial peptides are randomly generated.

<code>alleles</code> : the path of the file with all the alleles to be tested.

<code>out</code> : the path of the output file

<code>rollout</code> : the number of optimized peptides for each allele.

<code>path</code>: model path



To optimize specific peptides for given alleles with the trained PepPPO model , run

```
python ./code/test_RL.py --sample_rate 0.5 --peptides <test peptide path> --alleles ./data/test_alleles.txt --out <test result file> --rollout 1000 --path ./model/ppo_peptide.zip
```

<code>peptides</code> : the path of the file with all the alleles to be tested.
