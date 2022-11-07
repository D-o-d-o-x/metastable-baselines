# Metastable Baselines

<p align='center'>
  <img src='./icon.svg'>
</p>

During training of a RL-Agent we follow the gradient of the loss, which leads us to a minimum. In cases where the found minimum is merely a local minimum, this can be seen as a *false vacuum* in our loss space. Exploration mechanisms try to let our training procedure escape these *stable states*: Making them *metastable*. 

In order to archive this, this Repo contains some extensions for [Stable Baselines 3 by DLR-RM](https://github.com/DLR-RM/stable-baselines3)  
These extensions include:

- An implementation of ["Differentiable Trust Region Layers for Deep Reinforcement Learning" by Fabian Otto et al.](https://arxiv.org/abs/2101.09207)
- Support for Contextual Covariances
- Multiple parameterization strategies for the Covariance

The resulting algorithms can than be tested for their ability of exploration in the enviroments provided by [Project Columbus](https://git.dominik-roth.eu/dodox/Columbus)  

This Repo was created as part of my bachelor-thesis at ALR (KIT).

## Installation
#### (optional) Columbus for test.py and replay.py
Install [Project Columbus](https://git.dominik-roth.eu/dodox/Columbus) by following the instructions in the repo.  

#### Install dependency: Metastable Projections
Follow instructions for the [Public Version](https://git.dominik-roth.eu/dodox/metastable-projections-public) ([GitHub Mirror](https://github.com/D-o-d-o-x/metastable-projections-public)) / [Private Version](https://git.dominik-roth.eu/dodox/metastable-projections) ([GitHub Mirror](https://github.com/D-o-d-o-x/metastable-projections)).
The private version also requires ALR's ITPAL as a dependency. Only the private version supports KL Projections.

#### Install as a package
Then install this repo as a package:
```
pip install -e .
```

## License
Since this Repo is an extension to [Stable Baselines 3 by DLR-RM](https://github.com/DLR-RM/stable-baselines3), it contains some of it's code. SB3 is licensed under the [MIT-License](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE).
