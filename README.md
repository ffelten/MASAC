[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/ffelten/MASAC/blob/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MASAC

:warning: Work in progress, I did not extensively test the algorithms (especially the jax version).:warning:

Simple, yet useful [Jax](https://github.com/google/jax) and [Torch](https://pytorch.org/) Multi-Agent SAC for Parallel [PettingZoo](https://pettingzoo.farama.org/) environments.
It is assumed that the agents are homogeneous (actions and observations) and all have the same global reward.

The implementation is based on the SAC implementation from the excellent [cleanRL](https://github.com/vwxyzjn/cleanrl) repo.

## Multi-Agent features

Shared parameters:
  * Shared critic between all agents;
  * Shared actor (conditioned on agent ID).

## Install & run
```shell
poetry install
poetry run python masac/masac.py
```

## Citation
If you use this code for your research, please cite this using:

```bibtex
@misc{masac,
    author = {Florian Felten},
    title = {MASAC: A Multi-Agent Soft-Actor-Critic implementation for PettingZoo},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ffelten/MASAC}},
}
```
