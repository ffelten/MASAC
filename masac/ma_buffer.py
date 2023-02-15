"""Replay buffer for multi-agent reinforcement learning.

Stores the global and local observations of each agent (before and after action), along with actions, rewards, dones."""
import collections
from typing import Dict

import numpy as np
import torch as th

from utils import extract_agent_id


Experience = collections.namedtuple(
    "Experience",
    field_names=["global_obs", "local_obs", "joint_actions", "rewards", "next_global_obs", "next_local_obs", "terminateds"],
)


class MAReplayBuffer:
    """Multi-agent replay buffer for multi-agent reinforcement learning."""

    def __init__(
        self,
        global_obs_shape,
        local_obs_shape,
        action_dim,
        num_agents=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        """Initialize the replay buffer.

        Args:
            global_obs_shape: Shape of the global observations
            local_obs_shape: Shape of the locals observations
            action_dim: Dimension of the actions
            num_agents: Number of agents
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs_type = obs_dtype
        self.action_type = action_dtype
        self.global_obs = np.zeros((max_size,) + global_obs_shape, dtype=obs_dtype)
        self.local_obs = np.zeros((max_size, num_agents) + local_obs_shape, dtype=obs_dtype)
        self.next_global_obs = np.zeros((max_size,) + global_obs_shape, dtype=obs_dtype)
        self.next_local_obs = np.zeros((max_size, num_agents) + local_obs_shape, dtype=obs_dtype)
        self.joint_actions = np.zeros(
            (max_size, num_agents * action_dim), dtype=action_dtype
        )  # joint actions are flattened into a single vector
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.terminateds = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        global_obs: np.ndarray,
        local_obs: Dict[str, np.ndarray],
        joint_actions: np.ndarray,
        reward: float,
        next_global_obs: np.ndarray,
        next_local_obs: Dict[str, np.ndarray],
        terminated: bool,
    ):
        """Add a new experience to the buffer.

        Args:
            global_obs: Global observation
            local_obs: Local observation of each agent
            joint_actions: Actions of all agents (flattened into a single vector)
            reward: global reward
            next_global_obs: Next global observation
            next_local_obs: Next local observation of each agent
            terminated: Env is terminated or not
        """
        self.global_obs[self.ptr] = np.array(global_obs, dtype=self.obs_type).copy()
        self.next_global_obs[self.ptr] = np.array(next_global_obs, self.obs_type).copy()
        for agent_id, obs in local_obs.items():
            self.local_obs[self.ptr][extract_agent_id(agent_id)] = np.array(obs, dtype=self.obs_type).copy()
        for agent_id, obs in next_local_obs.items():
            self.next_local_obs[self.ptr][extract_agent_id(agent_id)] = np.array(obs, dtype=self.obs_type).copy()
        self.joint_actions[self.ptr] = np.array(joint_actions, dtype=self.action_type).copy()
        self.rewards[self.ptr] = np.array(reward, dtype=np.float32).copy()
        self.terminateds[self.ptr] = np.array(terminated).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, add_id_to_local_obs=False, device=None):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Whether to use CER
            to_tensor: Whether to convert the data to PyTorch tensors
            add_id_to_local_obs: Whether to add the agent id to the local observations
            device: Device to use

        Returns:
            An experience tuple:
                global_obs: Global observations (batch_size, global_obs_shape)
                local_obs:
                    Local observations of each agent (batch_size, num_agents, local_obs_shape)
                    (!) the dict is flattened into a vector
                    If add_id_to_local_obs is True, the local observations vectors are concatenated with the agent id
                joint_actions: Actions of all agents (batch_size, num_agents * action_dim)
                rewards: Rewards (batch_size,)
                next_global_obs: Next global observations (batch_size, global_obs_shape)
                next_local_obs: Next local observations of each agent (batch_size, num_agents, local_obs_shape) (!) the dict is flattened into a vector
                terminateds: Whether the episode is terminated or not (batch_size, 1)

        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience

        def flatten_local_obss(local_obs, inds):
            batch_local_obs = []
            for local_obs_ind in local_obs[inds]:
                local_obs_index = []
                for agent_id, local_obs in enumerate(local_obs_ind):
                    # This is very convenient for learning
                    if add_id_to_local_obs:
                        local_obs = np.concatenate((local_obs, np.array([agent_id], dtype=self.obs_type)))
                    if to_tensor:
                        local_obs_index.append(th.tensor(local_obs).to(device))
                    else:
                        local_obs_index.append(np.array(local_obs))
                if to_tensor:
                    batch_local_obs.append(th.stack(local_obs_index).to(device))
                else:
                    batch_local_obs.append(np.array(local_obs_index))
            if to_tensor:
                return th.stack(batch_local_obs).to(device)
            else:
                return np.array(batch_local_obs)

        if to_tensor:

            return Experience(
                global_obs=th.tensor(self.global_obs[inds]).to(device),
                local_obs=flatten_local_obss(self.local_obs, inds),
                joint_actions=th.tensor(self.joint_actions[inds]).to(device),
                rewards=th.tensor(self.rewards[inds]).to(device),
                next_global_obs=th.tensor(self.next_global_obs[inds]).to(device),
                next_local_obs=flatten_local_obss(self.next_local_obs, inds),
                terminateds=th.tensor(self.terminateds[inds]).to(device),
            )
        else:
            return Experience(
                global_obs=self.global_obs[inds],
                local_obs=flatten_local_obss(self.local_obs, inds),
                joint_actions=self.joint_actions[inds],
                rewards=self.rewards[inds],
                next_global_obs=self.next_global_obs[inds],
                next_local_obs=flatten_local_obss(self.next_local_obs, inds),
                terminateds=self.terminateds[inds],
            )

    def __len__(self):
        """Get the size of the buffer."""
        return self.size
