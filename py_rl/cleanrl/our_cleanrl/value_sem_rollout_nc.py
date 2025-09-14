# Text-semantic PPO for Polytopia
import os
import random
import time
from dataclasses import dataclass
from typing import cast

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
import einops

# add repo root so pol_env can be imported by name
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)



from sentence_transformers import SentenceTransformer
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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "hackmit"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
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
    num_minibatches: int = 4
    """the number of mini-batches"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Text embedding arguments
    text_model: str = "all-MiniLM-L6-v2"
    text_embed_dim: int = 384
    """sentence transformer model for action embeddings"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    epsilon: float = 0.1


def make_env(text_encoder, idx, capture_video, run_name):
    def thunk():
        base_env = TribesGymWrapper()
        if capture_video and idx == 0:
            print("recording video!")
            base_env = gym.wrappers.RecordVideo(base_env, f"videos/{run_name}", episode_trigger=lambda x: x % 10 == 0, video_length=64)
        env = gym.wrappers.RecordEpisodeStatistics(base_env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueModel(nn.Module):
    def __init__(self, envs, text_embed_dim):
        super().__init__()
        
        print(envs.single_observation_space.shape)
        obs_size = np.array(envs.single_observation_space.shape).prod()
        
        # Value function (doesn't need action info)
        self.obs_emb = layer_init(nn.Linear(obs_size, 1024))

        self.act_emb = layer_init(nn.Linear(text_embed_dim, 1024))

        self.value_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(1024),
                layer_init(nn.Linear(1024, 2048)),
                nn.ReLU(),
                layer_init(nn.Linear(2048, 1024)),
            ),
            nn.Sequential(
                nn.LayerNorm(1024),
                layer_init(nn.Linear(1024, 2048)),
                nn.ReLU(),
                layer_init(nn.Linear(2048, 1024)),
            ),
            nn.Sequential(
                nn.LayerNorm(1024),
                layer_init(nn.Linear(1024, 2048)),
                nn.ReLU(),
                layer_init(nn.Linear(2048, 1024)),
            ),
        ])
        
        self.final_norm = nn.LayerNorm(1024)
        self.value_head = layer_init(nn.Linear(1024, 1), std=1.0)
        
    def forward(self, x, act_emb):
        # x: (B, O,)
        # act_emb: (B, K, D)

        flattened = len(x.shape) == 2 and len(act_emb.shape) == 3
        if flattened:
            assert x.shape[0] == act_emb.shape[0]
            b_dim, obs_dim = x.shape
            b_dim, n_actions_dim, action_dim = act_emb.shape
            x = x.reshape(b_dim, 1, action_dim).repeat(1, n_actions_dim, 1)
            act_emb = act_emb.reshape(b_dim*n_actions_dim, action_dim)
            x = x.reshape(b_dim * n_actions_dim, obs_dim)
            print(x.shape, act_emb.shape)

        # Embed observation
        obs_features = torch.relu(self.obs_emb(x))
        
        # Embed action
        act_features = torch.relu(self.act_emb(act_emb))
        
        # Combine observation and action features
        combined = obs_features + act_features
        
        # Pass through residual blocks
        for layer in self.value_layers:
            residual = combined
            combined = layer(combined) + residual
            combined = torch.relu(combined)
        
        # Final normalization and value prediction
        combined = self.final_norm(combined)
        value = self.value_head(combined)

        if flattened:
            value = einops.rearrange(value, '(b a) d -> b a d', b=b_dim)

        return value


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

    # Initialize text encoder
    print("Loading text encoder...")
    text_encoder = SentenceTransformer(args.text_model)
    print(f"Text encoder loaded.")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(text_encoder, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = ValueModel(envs, args.text_embed_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Storage for action embeddings
    max_actions = envs.single_action_space.n
    action_embeddings_storage = torch.zeros((args.num_steps, args.num_envs, args.text_embed_dim)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("Starting training...")
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if iteration % 10 == 1:
            if args.capture_video:
                envs.envs[0].env.start_recording(video_name=f"step{iteration}")

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get current action embeddings for all environments
            current_embeddings = []
            selected_actions = []
            for i, env in enumerate(envs.envs):
                all_actions = env.unwrapped.tribes_env.list_actions()
                action_texts = [action['repr'] for action in all_actions]
                embeddings = torch.from_numpy(text_encoder.encode(action_texts)).to(device)
                current_embeddings.append(embeddings)

                expected_returns = agent(
                    next_obs[i],
                    embeddings,
                )

                if random.random() > args.epsilon:
                    selected_action = torch.argmax(expected_returns).item()
                else:
                    selected_action = random.randint(0, len(all_actions) - 1)
                
                selected_actions.append(selected_action)



                # 1. For each env index i has n_i valid actions
                # 2. For each n_i, forward each (state, action) through value model.
                # 3. Then select action as either just argmax, or epsilon greedy?



                # selected_actions.append(random.randint(0, len(all_actions) - 1))
            
            actions[step] = torch.tensor(selected_actions).to(device)
            #print(selected_actions[0])
            #print(current_embeddings[0][0])
            #print(torch.stack([current_embeddings[i][selected_actions[i]] for i in range(args.num_envs)], dim=0).to(device))
            action_embeddings_storage[step] = torch.stack([current_embeddings[i][selected_actions[i]] for i in range(args.num_envs)], dim=0).to(device)


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(selected_actions)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = 0
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)/30
        b_action_embeddings = action_embeddings_storage.reshape((-1, args.text_embed_dim))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            expected_ret = agent(
                b_obs[mb_inds], 
                b_action_embeddings[mb_inds]
            )
            loss = (expected_ret - b_returns[mb_inds]) ** 2
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        y_pred, y_true = expected_ret.detach().cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/y_pred_mean", y_pred.mean(), global_step)
        writer.add_scalar("train/y_true_mean", y_true.mean(), global_step)
        writer.add_scalar("train/y_pred_std", y_pred.std(), global_step)
        writer.add_scalar("train/y_true_std", y_true.std(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"step={iteration},loss={loss.item()}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
