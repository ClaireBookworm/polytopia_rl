# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    import pol_env.Tribes.py.register_env as register_env  # adjust if folder name differs
except Exception:
    _fpath = os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py")
    spec = importlib.util.spec_from_file_location("register_env", _fpath)
    register_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register_env)

from pol_env.Tribes.py.register_env import TribesGymWrapper  # adjust if folder name differs

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
    # env_id: str = "CartPole-v1"
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
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Semantic action arguments
    use_action_embeddings: bool = True
    """whether to use semantic action embeddings"""
    action_embed_dim: int = 16
    """dimension of action type embeddings"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class SemanticTribesWrapper(TribesGymWrapper):
    """Enhanced wrapper that provides semantic action information"""
    
    def __init__(self, level_file="levels/SampleLevel.csv"):
        super().__init__(level_file)
        
        # Define action type mapping
        self.action_types = {
            'RESEARCH_TECH': 0,
            'DECLARE_WAR': 1, 
            'SEND_STARS': 2,
            'END_TURN': 3,
            'MOVE': 4,
            'ATTACK': 5,
            'BUILD': 6,
            'BUILD_ROAD': 7,
            'CAPTURE': 8,
            'SPAWN': 9,
            'UPGRADE': 10,
            'UNKNOWN': 11  # fallback
        }
        self.num_action_types = len(self.action_types)
        
        # Cache for action metadata
        self._action_metadata = {}
        
    @property
    def action_space_n(self):
        """Get current number of available actions"""
        return self.tribes_env.action_space_n
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self._update_action_metadata()
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self._update_action_metadata()
        return obs, reward, done, truncated, info
        
    def _update_action_metadata(self):
        """Update cached action metadata"""
        actions = self.tribes_env.list_actions()
        self._action_metadata = {}
        
        for i, action_data in enumerate(actions):
            action_type = action_data.get('type', 'UNKNOWN')
            action_type_id = self.action_types.get(action_type, self.action_types['UNKNOWN'])
            
            self._action_metadata[i] = {
                'type_id': action_type_id,
                'type_name': action_type,
                'repr': action_data.get('repr', ''),
                'raw': action_data
            }
    
    def get_action_types(self, action_indices):
        """Get action type IDs for given action indices"""
        return [self._action_metadata.get(idx, {'type_id': self.action_types['UNKNOWN']})['type_id'] 
                for idx in action_indices]


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = SemanticTribesWrapper()
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = SemanticTribesWrapper()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SemanticAgent(nn.Module):
    def __init__(self, envs, use_action_embeddings=True, action_embed_dim=16):
        super().__init__()
        self.use_action_embeddings = use_action_embeddings
        
        obs_size = np.array(envs.single_observation_space.shape).prod()
        
        # Action type embedding layer
        if use_action_embeddings:
            # Get num_action_types from the actual environment
            env = envs.envs[0]
            while hasattr(env, 'env'):
                if hasattr(env, 'num_action_types'):
                    break
                env = env.env
            self.num_action_types = env.num_action_types
            self.action_embedding = nn.Embedding(self.num_action_types, action_embed_dim)
            # Combine observation with action type embeddings
            actor_input_size = obs_size + action_embed_dim
        else:
            actor_input_size = obs_size
            
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor network that outputs raw logits
        self.actor_base = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        if use_action_embeddings:
            # Action-aware head
            self.action_head = nn.Sequential(
                layer_init(nn.Linear(64 + action_embed_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.01),
            )
        else:
            # Standard action head
            self.action_head = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, valid_actions_mask=None, envs=None):
        # Get base features
        features = self.actor_base(x)
        
        if self.use_action_embeddings and envs is not None:
            # Get all possible action types for current state
            batch_size = x.shape[0]
            device = x.device
            
            # For each environment, get current action types
            all_logits = []
            
            for env_idx in range(batch_size):
                # Get the actual environment (unwrap gym wrappers)
                env = envs.envs[env_idx]
                while hasattr(env, 'env'):
                    if hasattr(env, 'action_space_n'):
                        break
                    env = env.env
                
                current_action_count = env.action_space_n
                current_actions = list(range(current_action_count))
                
                if len(current_actions) == 0:
                    # No valid actions, create dummy logits
                    logits = torch.full((envs.single_action_space.n,), float('-inf'), device=device)
                else:
                    # Get action types for current valid actions
                    action_types = env.get_action_types(current_actions)
                    action_type_tensor = torch.tensor(action_types, device=device)
                    
                    # Get embeddings for these action types
                    action_embeds = self.action_embedding(action_type_tensor)
                    
                    # Combine with observation features
                    env_features = features[env_idx:env_idx+1].expand(len(current_actions), -1)
                    combined = torch.cat([env_features, action_embeds], dim=1)
                    
                    # Get action preferences
                    action_scores = self.action_head(combined).squeeze(-1)
                    
                    # Create full logits tensor
                    logits = torch.full((envs.single_action_space.n,), float('-inf'), device=device)
                    for i, action_idx in enumerate(current_actions):
                        if action_idx < envs.single_action_space.n:
                            logits[action_idx] = action_scores[i]
                
                all_logits.append(logits)
            
            logits = torch.stack(all_logits)
        else:
            # Standard approach
            logits = self.action_head(features)
            
            # Apply action mask if provided
            if valid_actions_mask is not None:
                logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = SemanticAgent(envs, args.use_action_embeddings, args.action_embed_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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

            # ALGO LOGIC: action logic with semantic awareness
            with torch.no_grad():
                if args.use_action_embeddings:
                    action, logprob, _, value = agent.get_action_and_value(next_obs, envs=envs)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.use_action_embeddings:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds], envs=envs
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
