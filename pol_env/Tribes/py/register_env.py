import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv, make_default_env

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    def __init__(self, level_file="levels/SampleLevel.csv"):
        self.tribes_env = make_default_env()
        self.level_file = level_file
        self.render_mode = "rgb_array"        # Initialize the environment to get the actual action space size
        try:
            obs = self.tribes_env.reset(self.level_file, seed=42)
            
            # Use a fixed large action space to handle dynamic action counts
            # Most games will have fewer actions, but this provides a safe upper bound
            max_actions = 200  # Reasonable upper bound for this game
            self.action_space = gym.spaces.Discrete(max_actions)
            
            # Get observation dimensions from actual observation
            obs_array = self._dict_to_array(obs)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=obs_array.shape, 
                dtype=np.float32
            )
        except Exception as e:
            # Fallback to placeholders if initialization fails
            print(f"Warning: Could not initialize environment properly: {e}")
            self.action_space = gym.spaces.Discrete(200)  # reasonable default
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1000,), 
                dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        obs = self.tribes_env.reset(self.level_file, seed or 42)
        
        # Log action space info for debugging
        action_count = self.tribes_env.action_space_n
        print(f"Reset: Available actions = {action_count}")
        
        # convert your dict obs to numpy array here
        return self._dict_to_array(obs), {"valid_actions": action_count}
    
    def render(self, **kwargs):
        data = np.array(self.tribes_env.render("rgb_image"))
        return data

    def step(self, action):
        # Get current valid action count
        current_action_count = self.tribes_env.action_space_n
        
        # If action is invalid, map it to a valid action
        if action >= current_action_count:
            # Map large actions to valid range using modulo
            action = action % current_action_count
            
        obs, reward, done, info = self.tribes_env.step(action)
        
        # Add current action count to info for debugging
        info['valid_actions'] = current_action_count
        info['original_action'] = action
        
        return self._dict_to_array(obs), reward, done, False, info
    
    def _dict_to_array(self, obs_dict):
        # convert your complex dict observation to flat array
        # this is the key missing piece - you need to flatten your board state
        features = []
        
        board = obs_dict.get("board", {})
        
        # Extract terrain as flattened array
        terrain = board.get("terrain", [])
        if terrain:
            features.extend(np.array(terrain).flatten())
        
        # Extract unit IDs as flattened array  
        unit_ids = board.get("unitID", [])
        if unit_ids:
            features.extend(np.array(unit_ids).flatten())
            
        # Extract city IDs as flattened array
        city_ids = board.get("cityID", [])
        if city_ids:
            features.extend(np.array(city_ids).flatten())
            
        # Add tribe information
        tribe_info = obs_dict.get("tribe", {})
        if isinstance(tribe_info, dict):
            # Add some key tribe stats as features
            features.append(tribe_info.get("stars", 0))
            features.append(tribe_info.get("score", 0))
            features.append(len(tribe_info.get("citiesID", [])))
            features.append(tribe_info.get("nKills", 0))
            
        # Add game state info
        features.append(obs_dict.get("tick", 0))
        features.append(obs_dict.get("activeTribeID", 0))
        
        return np.array(features, dtype=np.float32)
    
    def close(self):
        """Close the underlying environment"""
        if hasattr(self.tribes_env, 'close'):
            self.tribes_env.close()

register(
    id="Tribes-v0",
    entry_point="pol_env.Tribes.py.register_env:TribesGymWrapper",
    max_episode_steps=1000,
)