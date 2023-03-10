[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/ffelten/MASAC/blob/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MASAC

:warning: Work in progress, suggestions are welcome. :warning:

Simple, yet useful Multi-Agent SAC for Parallel [PettingZoo](https://pettingzoo.farama.org/) environments.
It is assumed that the agents are homogeneous (actions and observations) and all have the same global reward.

The implementation is based on the SAC implementation from the excellent [cleanRL](https://github.com/vwxyzjn/cleanrl) repo.

## Multi-Agent features

Shared parameters:
  * Shared critic between all agents;
  * Shared actor (conditioned on agent ID).
