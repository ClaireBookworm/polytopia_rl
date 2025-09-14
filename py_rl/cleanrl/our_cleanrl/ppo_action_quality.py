# PPO with Action Quality Prediction
# Instead of just selecting actions by index, the model learns to evaluate the quality of each action
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import importlib.util
import sys

# add repo root (two levels up) so `pol_env` can be imported by name
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# try normal import first; fall back to loading by file path if needed
try:
    import pol_env.Tribes.py.register_env as register_env
except Exception:
    _fpath = os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py")
    spec = importlib.util.spec_from_file_location("register_env", _fpath)
    register_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register_env)

from pol_env.Tribes.py.register_env import TribesGymWrapper

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Tribes-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Action quality specific arguments
    action_encoding_dim: int = 32
    """dimension for action encoding"""
    quality_hidden_dim: int = 128
    """hidden dimension for action quality network"""
    top_k_actions: int = 10
    """number of top actions to consider for multi-step value prediction"""
    value_prediction_depth: int = 3
    """depth of value prediction (how many steps ahead to consider)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class ActionQualityTribesWrapper:
    """Wrapper that provides action quality evaluation capabilities"""
    
    def __init__(self, base_env):
        self.base_env = base_env
        self._current_actions = []
        self._action_space_size = None
        self._turn_count = 0
        
    def __getattr__(self, name):
        # Delegate all other attributes to base environment
        return getattr(self.base_env, name)
    
    @property
    def unwrapped(self):
        """Provide access to unwrapped environment"""
        return self.base_env.unwrapped if hasattr(self.base_env, 'unwrapped') else self.base_env
        
    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        self._turn_count = 0
        self._update_actions()
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.base_env.step(action)
        self._turn_count += 1
        self._update_actions()
        return obs, reward, done, truncated, info
        
    def _prune_actions(self, actions, max_send_per_turn=3):
        """Prune excessive SEND actions to reduce action space skew"""
        if self._turn_count < 10:  # Early game: focus on expansion/building
            # Filter out most SEND actions early
            pruned = []
            send_count = 0
            for action in actions:
                action_str = str(action.get('repr', ''))
                if 'SEND' in action_str:
                    if send_count < 1:  # Allow only 1 SEND early game
                        send_count += 1
                        pruned.append(action)
                else:
                    pruned.append(action)
            return pruned
        else:
            # Later game: allow more SENDs but still limit
            pruned = []
            send_count = 0
            for action in actions:
                action_str = str(action.get('repr', ''))
                if 'SEND' in action_str:
                    if send_count < max_send_per_turn:
                        send_count += 1
                        pruned.append(action)
                else:
                    pruned.append(action)
            return pruned
        
    def _update_actions(self):
        """Update current available actions"""
        try:
            raw_actions = self.base_env.tribes_env.list_actions()
            self._current_actions = self._prune_actions(raw_actions)
            if hasattr(self.base_env, 'action_space'):
                self._action_space_size = self.base_env.action_space.n
            else:
                self._action_space_size = len(self._current_actions)
        except:
            self._current_actions = []
            self._action_space_size = 0
            
    def get_current_actions(self) -> List[Dict[str, Any]]:
        """Get current available actions"""
        if not hasattr(self, '_current_actions'):
            self._update_actions()
        return self._current_actions
    
    def get_action_space_size(self) -> int:
        """Get current action space size"""
        if self._action_space_size is None:
            self._update_actions()
        return self._action_space_size or 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            base_env = gym.make(env_id, render_mode="rgb_array")
            base_env = gym.wrappers.RecordVideo(base_env, f"videos/{run_name}")
        else:
            base_env = gym.make(env_id)
        
        # Wrap with our action quality wrapper
        env = ActionQualityTribesWrapper(base_env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActionEncoder(nn.Module):
    """Encodes action information into fixed-size vectors"""
    
    def __init__(self, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim
        
    def encode_action(self, action_data: Dict[str, Any]) -> torch.Tensor:
        """Encode a single action into a vector"""
        # Simple action encoding based on action structure
        features = []
        
        # Action type encoding (12 main types we identified before)
        action_types = {
            'RESEARCH_TECH': 0, 'MOVE': 1, 'ATTACK': 2, 'BUILD': 3, 'END_TURN': 4,
            'DECLARE': 5, 'SEND': 6, 'CAPTURE': 7, 'SPAWN': 8, 'DISBAND': 9,
            'UPGRADE': 10, 'RECOVER': 11
        }
        
        action_str = str(action_data.get('repr', ''))
        action_type = 12  # default "other"
        for atype, idx in action_types.items():
            if atype in action_str:
                action_type = idx
                break
        
        # One-hot encode action type (13 dimensions)
        type_encoding = [0.0] * 13
        type_encoding[action_type] = 1.0
        features.extend(type_encoding)
        
        # Extract numeric features from action
        import re
        numbers = re.findall(r'\d+', action_str)
        
        # Position/coordinate features
        if len(numbers) >= 2:
            features.extend([float(numbers[0]) / 100.0, float(numbers[1]) / 100.0])
        else:
            features.extend([0.0, 0.0])
            
        # Unit/building ID features
        if len(numbers) >= 3:
            features.append(float(numbers[2]) / 100.0)
        else:
            features.append(0.0)
            
        # Additional action complexity
        features.append(len(action_str) / 100.0)  # Action description length
        features.append(len(numbers) / 10.0)      # Number of parameters
        
        # Pad or truncate to encoding_dim
        while len(features) < self.encoding_dim:
            features.append(0.0)
        features = features[:self.encoding_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def encode_actions(self, actions: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode a list of actions"""
        if not actions:
            # Return dummy encoding for empty action list
            return torch.zeros((1, self.encoding_dim))
        
        encoded = [self.encode_action(action) for action in actions]
        return torch.stack(encoded)


class ActionQualityAgent(nn.Module):
    """PPO Agent that predicts action quality using multi-step value prediction"""
    
    def __init__(self, envs, action_encoding_dim: int = 32, quality_hidden_dim: int = 128, 
                 top_k_actions: int = 10, value_prediction_depth: int = 3):
        super().__init__()
        
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        self.action_encoder = ActionEncoder(action_encoding_dim)
        self.top_k_actions = top_k_actions
        self.value_prediction_depth = value_prediction_depth
        
        # Value function (unchanged)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Multi-step value predictor - predicts value changes over next N steps
        # Takes observation + action encoding -> multi-step value prediction
        self.multistep_value_net = nn.Sequential(
            layer_init(nn.Linear(obs_shape + action_encoding_dim, quality_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(quality_hidden_dim, quality_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(quality_hidden_dim, quality_hidden_dim)),
            nn.Tanh(),
            # Output 10 values for next 10 steps
            layer_init(nn.Linear(quality_hidden_dim, 10), std=0.01),
        )
        
        # Action quality predictor that combines immediate and multi-step predictions
        self.quality_fusion_net = nn.Sequential(
            layer_init(nn.Linear(obs_shape + action_encoding_dim + 10, quality_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(quality_hidden_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        
        # For compatibility with PPO training loop
        self.max_actions = 500  # reasonable upper bound for action space
        
    def get_value(self, x):
        return self.critic(x)
    
    def predict_multistep_values(self, obs: torch.Tensor, action_encodings: torch.Tensor) -> torch.Tensor:
        """Predict values for the next 10 steps after taking each action"""
        batch_size = obs.shape[0]
        num_actions = action_encodings.shape[0]
        
        # Expand observation to match each action
        obs_expanded = obs.unsqueeze(1).expand(batch_size, num_actions, -1)  # (batch, actions, obs_dim)
        action_expanded = action_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, actions, action_dim)
        
        # Concatenate obs + action encoding
        combined = torch.cat([obs_expanded, action_expanded], dim=-1)  # (batch, actions, obs_dim + action_dim)
        
        # Reshape for batch processing
        combined_flat = combined.view(-1, combined.shape[-1])  # (batch * actions, obs_dim + action_dim)
        
        # Get multi-step value predictions
        multistep_values = self.multistep_value_net(combined_flat)  # (batch * actions, 10)
        multistep_values = multistep_values.view(batch_size, num_actions, 10)  # (batch, actions, 10)
        
        return multistep_values
    
    def evaluate_action_quality(self, obs: torch.Tensor, action_encodings: torch.Tensor) -> torch.Tensor:
        """Evaluate quality scores using multi-step value prediction"""
        batch_size = obs.shape[0]
        num_actions = action_encodings.shape[0]
        
        # Get multi-step value predictions
        multistep_values = self.predict_multistep_values(obs, action_encodings)  # (batch, actions, 10)
        
        # Compute discounted average over next 10 steps
        gamma_weights = torch.tensor([0.99 ** i for i in range(10)], device=obs.device)
        gamma_weights = gamma_weights / gamma_weights.sum()  # Normalize weights
        
        # Weighted average of predicted values
        discounted_avg_value = torch.sum(multistep_values * gamma_weights.unsqueeze(0).unsqueeze(0), dim=2)  # (batch, actions)
        
        # Expand inputs for fusion network
        obs_expanded = obs.unsqueeze(1).expand(batch_size, num_actions, -1)  # (batch, actions, obs_dim)
        action_expanded = action_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, actions, action_dim)
        
        # Concatenate obs + action + multistep predictions for final quality assessment
        fusion_input = torch.cat([
            obs_expanded, 
            action_expanded, 
            multistep_values
        ], dim=-1)  # (batch, actions, obs_dim + action_dim + 10)
        
        # Reshape for batch processing
        fusion_input_flat = fusion_input.view(-1, fusion_input.shape[-1])  # (batch * actions, obs_dim + action_dim + 10)
        
        # Get final quality scores
        quality_scores = self.quality_fusion_net(fusion_input_flat)  # (batch * actions, 1)
        quality_scores = quality_scores.view(batch_size, num_actions)  # (batch, actions)
        
        return quality_scores
    
    def get_action_and_value(self, obs: torch.Tensor, available_actions: List[List[Dict[str, Any]]] = None, action=None):
        """Get action and value, using action quality evaluation"""
        
        if available_actions is None:
            # Fallback to dummy action selection if no actions provided
            dummy_logits = torch.zeros(obs.shape[0], self.max_actions, device=obs.device)
            probs = Categorical(logits=dummy_logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
        
        batch_size = obs.shape[0]
        action_logits_list = []
        selected_actions = []
        log_probs = []
        entropies = []
        
        for i in range(batch_size):
            env_actions = available_actions[i] if i < len(available_actions) else []
            
            if not env_actions:
                # No actions available, use dummy
                logits = torch.zeros(1, device=obs.device)
                selected_action = 0
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(0, device=obs.device))
                entropy = dist.entropy()
            else:
                # Encode all available actions
                action_encodings = self.action_encoder.encode_actions(env_actions).to(obs.device)
                
                # Get quality scores for this observation
                obs_single = obs[i:i+1]  # Keep batch dimension
                quality_scores = self.evaluate_action_quality(obs_single, action_encodings)
                logits = quality_scores.squeeze(0)  # Remove batch dimension
                
                # Create distribution from quality scores
                dist = Categorical(logits=logits)
                
                if action is None:
                    # SAMPLE from the policy (not argmax) for exploration
                    action_tensor = dist.sample()
                    selected_action = action_tensor.item()
                    log_prob = dist.log_prob(action_tensor)
                else:
                    # During training: compute log prob of given action
                    action_idx = action[i].item() if action[i] < len(env_actions) else 0
                    selected_action = action_idx
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=obs.device))
                
                entropy = dist.entropy()
            
            action_logits_list.append(logits)
            selected_actions.append(selected_action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        if action is None:
            action = torch.tensor(selected_actions, device=obs.device)
        
        # Stack the individual log probs and entropies
        stacked_log_probs = torch.stack(log_probs)
        stacked_entropies = torch.stack(entropies)
        
        return action, stacked_log_probs, stacked_entropies, self.critic(obs)


class ActionQualityVectorEnv:
    """Vector environment wrapper that collects actions from all environments"""
    
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = envs.num_envs
        
    def __getattr__(self, name):
        return getattr(self.envs, name)
    
    def get_all_available_actions(self):
        """Get available actions from all environments"""
        all_actions = []
        for i in range(self.num_envs):
            try:
                env = self.envs.envs[i].env  # Access underlying wrapper
                if hasattr(env, 'get_current_actions'):
                    actions = env.get_current_actions()
                else:
                    actions = []
                all_actions.append(actions)
            except:
                all_actions.append([])
        return all_actions


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    base_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    envs = ActionQualityVectorEnv(base_envs)
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = ActionQualityAgent(
        envs, 
        action_encoding_dim=args.action_encoding_dim,
        quality_hidden_dim=args.quality_hidden_dim,
        top_k_actions=args.top_k_actions,
        value_prediction_depth=args.value_prediction_depth
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Storage for action contexts (needed for training)
    action_contexts = [[None for _ in range(args.num_envs)] for _ in range(args.num_steps)]
    
    # Storage for multi-step returns (for training the multi-step predictor)
    multistep_returns = torch.zeros((args.num_steps, args.num_envs, 10)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("Starting training with action quality evaluation...")
    
    # Track action type preferences
    action_type_counts = {
        'RESEARCH_TECH': 0, 'MOVE': 0, 'ATTACK': 0, 'BUILD': 0, 'END_TURN': 0,
        'DECLARE': 0, 'SEND': 0, 'CAPTURE': 0, 'SPAWN': 0, 'OTHER': 0
    }
    total_actions_taken = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic with quality evaluation
            with torch.no_grad():
                # Get available actions from all environments
                available_actions = envs.get_all_available_actions()
                
                # Store action contexts for training
                for env_idx in range(args.num_envs):
                    action_contexts[step][env_idx] = available_actions[env_idx] if env_idx < len(available_actions) else []
                
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, 
                    available_actions=available_actions
                )
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            # Track action type statistics
            available_actions = envs.get_all_available_actions()
            for env_idx, env_action_idx in enumerate(action.cpu().numpy()):
                if env_idx < len(available_actions) and env_action_idx < len(available_actions[env_idx]):
                    action_info = available_actions[env_idx][env_action_idx]
                    action_str = str(action_info.get('repr', ''))
                    
                    # Classify action type
                    action_type = 'OTHER'
                    for atype in action_type_counts.keys():
                        if atype in action_str:
                            action_type = atype
                            break
                    
                    action_type_counts[action_type] += 1
                    total_actions_taken += 1
            
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            # Compute multi-step returns for training the multi-step predictor
            for t in range(args.num_steps):
                for step_ahead in range(10):  # Look 10 steps ahead
                    future_step = t + step_ahead
                    if future_step < args.num_steps:
                        # Use actual future rewards
                        discounted_reward = rewards[future_step] * (args.gamma ** step_ahead)
                        multistep_returns[t, :, step_ahead] = discounted_reward
                    else:
                        # Use bootstrapped value for steps beyond rollout
                        remaining_steps = future_step - args.num_steps + 1
                        discounted_value = next_value.squeeze(0) * (args.gamma ** step_ahead)
                        multistep_returns[t, :, step_ahead] = discounted_value
            
            # Standard GAE calculation
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_multistep_returns = multistep_returns.reshape(-1, 10)
        
        # Flatten action contexts
        b_action_contexts = []
        for step in range(args.num_steps):
            for env in range(args.num_envs):
                b_action_contexts.append(action_contexts[step][env])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Reconstruct action probabilities using stored contexts
                mb_action_contexts = [b_action_contexts[i] for i in mb_inds]
                
                # Compute new log probs and entropy for each item in minibatch
                newlogprob_list, entropy_list, newvalue_list = [], [], []
                multistep_loss_list = []
                
                for idx_in_mb, global_idx in enumerate(mb_inds):
                    obs_single = b_obs[global_idx:global_idx+1]  # (1, obs_dim)
                    action_single = b_actions.long()[global_idx].item()
                    action_context = mb_action_contexts[idx_in_mb] or []
                    actual_multistep_returns = b_multistep_returns[global_idx]  # (10,)
                    
                    if not action_context:
                        # Handle empty action context
                        dummy_logits = torch.zeros(1, device=device)
                        dist = Categorical(logits=dummy_logits)
                        newlogprob_list.append(dist.log_prob(torch.tensor(0, device=device)))
                        entropy_list.append(dist.entropy())
                        multistep_loss_list.append(torch.tensor(0.0, device=device))
                    else:
                        # Encode actions and get quality scores
                        action_encodings = agent.action_encoder.encode_actions(action_context).to(device)
                        quality_scores = agent.evaluate_action_quality(obs_single, action_encodings)
                        logits = quality_scores.squeeze(0)  # (num_actions,)
                        
                        dist = Categorical(logits=logits)
                        if action_single < len(action_context):
                            newlogprob_list.append(dist.log_prob(torch.tensor(action_single, device=device)))
                            
                            # Compute multi-step prediction loss for the selected action
                            selected_action_encoding = action_encodings[action_single:action_single+1]  # (1, action_dim)
                            predicted_multistep = agent.predict_multistep_values(obs_single, selected_action_encoding)  # (1, 1, 10)
                            predicted_multistep = predicted_multistep.squeeze()  # (10,)
                            
                            # MSE loss between predicted and actual multi-step returns
                            multistep_loss = nn.functional.mse_loss(predicted_multistep, actual_multistep_returns)
                            multistep_loss_list.append(multistep_loss)
                        else:
                            # Action index out of bounds, use dummy
                            newlogprob_list.append(dist.log_prob(torch.tensor(0, device=device)))
                            multistep_loss_list.append(torch.tensor(0.0, device=device))
                        entropy_list.append(dist.entropy())
                    
                    # Get value for this observation
                    newvalue_list.append(agent.get_value(obs_single).squeeze())
                
                newlogprob = torch.stack(newlogprob_list)
                entropy = torch.stack(entropy_list)
                newvalue = torch.stack(newvalue_list)
                multistep_loss = torch.stack(multistep_loss_list).mean()
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                
                # Total loss includes multi-step prediction loss
                multistep_coef = 0.1  # Coefficient for multi-step prediction loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + multistep_coef * multistep_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/multistep_loss", multistep_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Report action type preferences every 10 iterations
        if iteration % 10 == 0:
            print(f"\n=== Iteration {iteration} Action Quality Analysis ===")
            if total_actions_taken > 0:
                print("Action type preferences (% of actions taken):")
                for action_type, count in action_type_counts.items():
                    percentage = (count / total_actions_taken) * 100
                    print(f"  {action_type}: {percentage:.1f}% ({count} actions)")
                
                # Log to tensorboard
                for action_type, count in action_type_counts.items():
                    percentage = (count / total_actions_taken) * 100
                    writer.add_scalar(f"action_types/{action_type}_percentage", percentage, global_step)
            
            # Training progress indicators
            print(f"Policy loss: {pg_loss.item():.6f}")
            print(f"Value loss: {v_loss.item():.6f}")
            print(f"Entropy: {entropy_loss.item():.6f}")
            print(f"Multi-step loss: {multistep_loss.item():.6f}")
            print(f"Approx KL: {approx_kl.item():.6f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Total actions taken: {total_actions_taken}")
            print("=" * 50)

    envs.close()
    writer.close()
    
    print(f"\n🎉 TRAINING COMPLETE - ACTION QUALITY PPO RESULTS")
    print("=" * 60)
    print(f"Total timesteps: {global_step:,}")
    print(f"Total training time: {time.time() - start_time:.1f} seconds")
    
    if total_actions_taken > 0:
        print(f"\nFINAL ACTION TYPE ANALYSIS:")
        print(f"Total actions taken: {total_actions_taken:,}")
        print("\nAction type preferences:")
        
        # Sort by frequency
        sorted_actions = sorted(action_type_counts.items(), key=lambda x: x[1], reverse=True)
        for action_type, count in sorted_actions:
            percentage = (count / total_actions_taken) * 100
            print(f"  {action_type:<12}: {percentage:5.1f}% ({count:,} actions)")
        
        # Analyze strategy
        print(f"\nSTRATEGY ANALYSIS:")
        research_pct = (action_type_counts['RESEARCH_TECH'] / total_actions_taken) * 100
        combat_pct = (action_type_counts['ATTACK'] / total_actions_taken) * 100
        expansion_pct = (action_type_counts['MOVE'] / total_actions_taken) * 100
        building_pct = (action_type_counts['BUILD'] / total_actions_taken) * 100
        
        print(f"  Research focus: {research_pct:.1f}%")
        print(f"  Combat focus: {combat_pct:.1f}%") 
        print(f"  Expansion focus: {expansion_pct:.1f}%")
        print(f"  Building focus: {building_pct:.1f}%")
        
        if research_pct > 20:
            print("  → Strategy: Technology-focused")
        elif combat_pct > 15:
            print("  → Strategy: Military-focused") 
        elif expansion_pct > 30:
            print("  → Strategy: Expansion-focused")
        elif building_pct > 25:
            print("  → Strategy: Development-focused")
        else:
            print("  → Strategy: Balanced")
    
    print(f"\nTensorBoard logs saved to: runs/{run_name}")
    print(f"To view results: tensorboard --logdir runs/{run_name}")
    print("=" * 60)
