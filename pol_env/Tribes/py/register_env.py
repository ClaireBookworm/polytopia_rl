import gymnasium as gym
from numpy import np
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    def __init__(self, level_file="levels/SampleLevel.csv"):
        self.tribes_env = TribesGymEnv()
        self.level_file = level_file
        
        # define spaces - you'll need to figure these out from your obs
        self.action_space = gym.spaces.Discrete(1)  # placeholder
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(100,), dtype=float)  # placeholder
    
    def reset(self, seed=None, options=None):
        obs = self.tribes_env.reset(self.level_file, seed or 42)
        # convert your dict obs to numpy array here
        return self._dict_to_array(obs), {}
    
    def step(self, action):
        obs, reward, done, info = self.tribes_env.step(action)
        return self._dict_to_array(obs), reward, done, False, info
    
    def _dict_to_array(self, obs_dict):
        # convert your complex dict observation to flat array
        # this is the key missing piece - you need to flatten your board state
        board = obs_dict.get("board", {})
        terrain = board.get("terrain", [[]])
        # flatten terrain + other features into single array
        return np.array(terrain).flatten()[:100]  # truncate to fixed size

register(
    id="Tribes-v0",
    entry_point="pol_env.Tribes.py.register_env:TribesGymWrapper",
    max_episode_steps=1000,
)