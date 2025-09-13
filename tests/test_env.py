#!/usr/bin/env python3

import sys
import os

# Add the repo root to the path
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import gymnasium as gym
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    
    print("Testing TribesGymWrapper...")
    
    # Test direct instantiation
    env = TribesGymWrapper()
    print(f"✓ Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")
    print(f"Reset info: {info}")
    
    # Test several steps with different action values
    for i in range(5):
        action = env.action_space.sample()  # Random action from full space
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.3f}, done={done}, valid_actions={info.get('valid_actions', 'N/A')}")
        if done:
            print("Episode ended, resetting...")
            obs, info = env.reset()
    
    env.close()
    print("✓ Environment closed successfully")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
