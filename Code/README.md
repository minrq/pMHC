## Peptide RL

### 1. Overview

<code>src</code>  source code

<code>script</code> commands used for training and testing

<code>data</code> data used for training

<code>model</code> trained RL model

<code>stable_baselines3</code>  code from the package "stable_baseline3". Our implementation of RL model is based on this package.

<code>baseline</code> code for baselines including MCTS, VAE with backpropagation and VAE with bayesian optimization.



### 2. Usage

The reward of our RL model for peptide generation is the predicted presentation score from MHCflurry-2.0.

Note that MHCflurry-2.0 is based on keras with tensorflow as its backend.

To accelerate the training, we ran multiple instances of peptide environment in parallel using subprocesses. Each environment will run MHCflurry-2.0 with cpu independently. We found that the implementation with multiple environments running in parallel is much more efficient than the single environment. In our experiment, we used 30 environments.

