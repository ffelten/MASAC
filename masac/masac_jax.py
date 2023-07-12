"""This implementation is based on https://github.com/Howuhh/sac-n-jax/ and
 cleanRL jax implementation: https://github.com/vwxyzjn/cleanrl/blob/15c30c8f91326f2f441dd28b5705b30b05450899/cleanrl/sac_continuous_action_jax.py
 """

import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Dict, Tuple

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import wandb
from etils import epath
from flax.training.train_state import TrainState
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.env import ObsType
from tqdm import tqdm

from ma_buffer import Experience, MAReplayBuffer


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="MASAC",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="florian-felten",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e4),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
                        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
                        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--num-critics", type=int, default=2, help="Number of Q networks used in ensemble.")
    parser.add_argument("--policy-frequency", type=int, default=1,
                        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,  # Denis Yarats' implementation delays this by 2.
                        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    args = parser.parse_args()
    # fmt: on
    return args


# Networks stuff
def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


class Actor(nn.Module):
    """Actor Network for MASAC, it takes the local state of each agent and its id and returns a continuous action."""

    action_dim: int
    hidden_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        # local state, id -> ... -> action_mean, action_std
        network = nn.Sequential(
            [
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
            ]
        )
        mean_layer = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)
        )
        log_std_layer = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = network(local_obs_and_id)
        mean, log_std = mean_layer(trunk), log_std_layer(trunk)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class Critic(nn.Module):
    """Critic network for the MASAC algorithm. Takes a global state and joint acton as input and returns a Q-value"""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_state_and_joint_actions: jnp.ndarray):
        # Global state, joint action -> ... -> Q value
        network = nn.Sequential(
            [
                nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)),
            ]
        )
        out = network(global_state_and_joint_actions).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    """Parallelize the ensemble of Q Networks used in SAC (2 in the original SAC implementation)"""

    hidden_dim: int = 256
    num_critics: int = 2

    @nn.compact
    def __call__(self, global_state_and_joint_actions):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim)(global_state_and_joint_actions)
        return q_values


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict = None


@partial(jax.jit, static_argnames="actor_module")
def sample_action(
    actor_module: Actor,
    actor_state: TrainState,
    local_observations_and_ids: jnp.ndarray,
    key: jax.random.KeyArray,
) -> jnp.array:
    """Sample an action from the actor network then feed it to a gaussian distribution to get a continuous action.

    Args:
        actor_module: Actor network.
        actor_state: Actor network parameters.
        local_observations_and_ids: Local observations and agent ids.
        key: JAX random key.

    Returns: A tuple of (action, key).
    """
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor_module.apply(actor_state.params, local_observations_and_ids)
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    action = jnp.tanh(gaussian_action)
    return action, key


@partial(jax.jit, static_argnames="actor_module")
def sample_action_for_agent(actor_module: Actor, actor_state: TrainState, obs: jnp.ndarray, agent_id, key):
    """Samples an action from the policy given an observation and an agent_id. This is vmapped to get all actions at each timestep."""
    obs_with_ids = jnp.append(obs[agent_id], agent_id)
    act, key = sample_action(actor_module, actor_state, obs_with_ids, key)
    return act, key


@jax.jit
def sample_action_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    subkey: jax.random.KeyArray,
):
    """Same as above except it returns the log prob as well (for learning).

    Args:
        mean: Mean of the Gaussian distribution.
        log_std: Log standard deviation of the Gaussian distribution.
        subkey: JAX random key.

    Returns: A tuple of (action, log_prob).
    """
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    log_prob = -0.5 * ((gaussian_action - mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
    log_prob = log_prob.sum(axis=1)
    action = jnp.tanh(gaussian_action)
    log_prob -= jnp.sum(jnp.log((1 - action**2) + 1e-6), 1)
    return action, log_prob


@partial(jax.jit, static_argnames="actor_module")
def sample_action_and_log_prob_for_agent(
    actor_module: Actor,
    actor_params: flax.core.FrozenDict,
    local_obs_and_id_for_agent: jnp.ndarray,
    subkey: jax.random.KeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the actions and log probabilities for a single agent. This is vmapped for each agent at each timestep."""
    mean, log_std = actor_module.apply(actor_params, local_obs_and_id_for_agent)
    actions, next_log_prob = sample_action_and_log_prob(mean, log_std, subkey)
    return actions, next_log_prob


def normalize_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
    """Rescale the action from [low, high] to [-1, 1]."""
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def denormalize_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
    """Rescale the action from [-1, 1] to [low, high]."""
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


@partial(jax.jit, static_argnames="tau")
def soft_update(tau: float, critic_state: RLTrainState) -> RLTrainState:
    """Update the target parameters of the critic network using Polyak averaging."""
    critic_state = critic_state.replace(
        target_params=optax.incremental_update(critic_state.params, critic_state.target_params, tau)
    )
    return critic_state


def main():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # INITIALISATION
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    single_observation_space = env.observation_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert isinstance(single_observation_space, gym.spaces.Box), "only continuous observation space is supported"

    # SEEDING AND GETTING ALL DIMENSIONS OF THE ENV
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(seed=args.seed)
    key, actor_key, critic_key, ent_key = jax.random.split(key, 4)
    init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
    init_local_state_and_id = jnp.append(init_local_state, jnp.array([0]))  # add a fake id to init the actor net
    init_action = jnp.asarray(env.action_space(env.unwrapped.agents[0]).sample())
    num_agents = env.num_agents
    init_joint_action = init_action.repeat(num_agents)  # fake joint action to init the critic networks
    init_global_state = jnp.asarray(env.state())

    # NETWORKS INIT
    actor_module = Actor(action_dim=np.prod(single_action_space.shape))
    actor_state = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_local_state_and_id),
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    critic_module = EnsembleCritic(num_critics=args.num_critics)
    init_global_state_and_joint_actions = jnp.append(init_global_state, init_joint_action)
    critic_state = RLTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init({"params": critic_key}, init_global_state_and_joint_actions),
        target_params=critic_module.init({"params": critic_key}, init_global_state_and_joint_actions),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    ent_coef = EntropyCoef(ent_coef_init=1.0)
    target_entropy = -np.prod(single_action_space.shape).astype(np.float32)
    ent_coef_state = TrainState.create(
        apply_fn=ent_coef.apply,
        params=ent_coef.init(ent_key)["params"],
        tx=optax.adam(learning_rate=args.q_lr),
    )

    # Define update functions here to limit the need for static argname
    @jax.jit
    def update_critic(
        actor_state: TrainState,
        critic_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        global_obs: np.ndarray,
        joint_actions: np.ndarray,
        next_global_obs: np.ndarray,
        next_local_obs_and_id: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        key: jax.random.KeyArray,
    ):
        """Returns the updated critic parameters.

        Args:
            actor_state: Current actor parameters.
            critic_state: Current critic parameters.
            ent_coef_value: Current entropy coefficient value. (alpha)
            global_obs: Global observations from replay buffer. Shape: (batch_size, global_obs_dim)
            joint_actions: Joint actions from replay buffer. Shape: (batch_size, joint_action_dim)
            next_global_obs: Next global observations from replay buffer. Shape: (batch_size, global_obs_dim)
            next_local_obs_and_id: Next local observations from replay buffer. Shape: (batch_size, num_agents, local_obs_dim + 1)
            rewards: Rewards from replay buffer. Shape: (batch_size,)
            terminateds: Terminateds signals from replay buffer. Shape: (batch_size,)
            key: Jax random key.
        """
        key, subkey = jax.random.split(key, 2)

        # Sample next actions and log probs for each agent (in parallel)
        next_state_actions, next_log_prob = jax.vmap(sample_action_and_log_prob_for_agent, in_axes=(None, None, 1, None))(
            actor_module, actor_state.params, jnp.array(next_local_obs_and_id), subkey
        )

        joint_next_state_actions = next_state_actions.reshape((args.batch_size, single_action_space.shape[0] * num_agents))
        next_log_prob = next_log_prob.sum(axis=0)  # TODO: check if this is correct
        global_state_and_joint_actions = jnp.concatenate((next_global_obs, joint_next_state_actions), axis=1)
        # critic next values is based on the target params!
        critic_next_values = critic_module.apply(critic_state.target_params, global_state_and_joint_actions)
        next_q_values = jnp.min(critic_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob
        next_q_values = next_q_values.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - terminateds.reshape(-1, 1)) * args.gamma * next_q_values

        def mse_loss(params: flax.core.FrozenDict):
            global_obs_and_joint_actions = jnp.concatenate((global_obs, joint_actions), axis=1)
            # shape is (n_critics, batch_size, 1)
            current_q_values = critic_module.apply(params, global_obs_and_joint_actions)
            current_q_values = current_q_values.reshape((args.num_critics, args.batch_size, 1))
            # mean over the batch and then sum for each critic
            critic_loss = 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()
            return critic_loss, current_q_values.mean()

        (critic_loss_value, critic_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)

        return (
            critic_state,
            (critic_loss_value, critic_values),
            key,
        )

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        critic_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        local_obs_and_ids: np.ndarray,
        global_obs: np.ndarray,
        key: jax.random.KeyArray,
    ):
        """Returns the updated actor state and loss info.

        Args:
            actor_state: Current actor state.
            critic_state: Current critic state.
            ent_coef_value: Current entropy coefficient value. (alpha)
            local_obs_and_ids: Local observations and agent ids for all agents. Shape is (batch_size, num_agents, obs_dim + 1)
            global_obs: Global observations. Shape is (batch_size, global_obs_dim)
            key: Random key.

        """
        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            actions, log_prob = jax.vmap(sample_action_and_log_prob_for_agent, in_axes=(None, None, 1, None))(
                actor_module, params, jnp.array(local_obs_and_ids), subkey
            )
            joint_actions = actions.reshape((args.batch_size, single_action_space.shape[0] * num_agents))
            log_prob = log_prob.sum(axis=0)
            global_obs_and_joint_actions = jnp.concatenate((global_obs, joint_actions), axis=1)
            critic_pi = critic_module.apply(critic_state.params, global_obs_and_joint_actions)
            # Take min among all critics
            min_critic_pi = jnp.min(critic_pi, axis=0)
            actor_loss = (ent_coef_value * log_prob - min_critic_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, critic_state, actor_loss_value, key, entropy

    @jax.jit
    def update_temperature(ent_coef_state: TrainState, entropy: float):
        """Returns the updated entropy coefficient state. (alpha)"""

        def temperature_loss(params):
            ent_coef_value = ent_coef.apply({"params": params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    single_observation_space.dtype = np.float32
    rb = MAReplayBuffer(
        global_obs_shape=env.state().shape,
        local_obs_shape=single_observation_space.shape,
        action_dim=single_action_space.shape[0],
        num_agents=env.max_num_agents,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    # !! Limitation of MASAC is that we assume the agent ids are from 0 to n-1
    agent_ids: jnp.ndarray = jnp.arange(num_agents)
    global_return = 0.0
    global_obs: np.ndarray = env.state()

    # Display progress bar if available
    generator = tqdm(range(args.total_timesteps)) if tqdm is not None else range(args.total_timesteps)
    for global_step in generator:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions: Dict[str, np.ndarray] = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions: Dict[str, np.ndarray] = {}
            obs_array = jnp.array([obs[agent] for agent in env.agents])
            acts, keys = jax.vmap(sample_action_for_agent, in_axes=(None, None, None, 0, None))(
                actor_module, actor_state, obs_array, agent_ids, key
            )
            # TODO split into subkeys?
            key = keys[0]

            # Construct the dict of actions for PZ
            for agent_id, act in zip(env.agents, acts):
                act = np.array(act)
                # Clip due to numerical instability
                act = np.clip(act, -1, 1)
                # Rescale to proper domain when using squashing
                act = denormalize_action(single_action_space, act)
                actions[agent_id] = act

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs: Dict[str, ObsType]
        rewards: Dict[str, float]
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # TODO PZ does not support that yet
        # if truncated:
        #     real_next_obs = infos["final_observation"].copy()

        # Scales the actions before storing in the replay buffer
        scaled_actions = {k: normalize_action(single_action_space, act) for k, act in actions.items()}

        rb.add(
            global_obs=global_obs,
            local_obs=obs,
            joint_actions=np.array(list(scaled_actions.values())).flatten(),
            reward=np.array(list(rewards.values())).sum(),
            next_global_obs=env.state(),
            next_local_obs=real_next_obs,
            terminated=terminated,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_return += sum(rewards.values())
        global_obs = env.state()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data: Experience = rb.sample(batch_size=args.batch_size, add_id_to_local_obs=True)

            ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

            critic_state, (critic_loss_value, critic_values), key = update_critic(
                actor_state=actor_state,
                critic_state=critic_state,
                ent_coef_value=ent_coef_value,
                global_obs=data.global_obs,
                joint_actions=data.joint_actions,
                next_global_obs=data.next_global_obs,
                next_local_obs_and_id=data.next_local_obs,
                rewards=data.rewards,
                terminateds=data.terminateds,
                key=key,
            )

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                (actor_state, critic_state, actor_loss_value, key, entropy) = update_actor(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    ent_coef_value=ent_coef_value,
                    local_obs_and_ids=data.local_obs,
                    global_obs=data.global_obs,
                    key=key,
                )

                ent_coef_state, ent_coef_loss = update_temperature(ent_coef_state, entropy)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                critic_state = soft_update(args.tau, critic_state)

            if global_step % 100 == 0 and args.track:
                to_log = {
                    "losses/critic_values": critic_values.mean().item(),
                    "losses/critic_loss": critic_loss_value.item(),
                    "losses/actor_loss": actor_loss_value.item(),
                    "losses/alpha": ent_coef_value.item(),
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "losses/alpha_loss": ent_coef_loss.item(),
                    "global_step": global_step,
                }
                wandb.log(to_log, step=global_step)

        if terminated or truncated:
            obs, info = env.reset()
            if args.track:
                wandb.log({"charts/return": global_return, "global_step": global_step}, step=global_step)
            global_return = 0.0
            global_obs = env.state()

    # SAVING MODEL PARAMETERS
    directory = epath.Path("trained_model")
    actor_dir = directory / "actor"
    print("Saving actor to ", actor_dir)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    ckptr.save(actor_dir, actor_state, force=True)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
