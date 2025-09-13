#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import gymnasium as gym
from pol_env.Tribes.py.register_env import TribesGymWrapper

# Test gym registration
print("Testing gym.make...")
env = gym.make("Tribes-v0")
obs, info = env.reset()
print(f"✓ gym.make works, obs shape: {obs.shape}")

# Test with wrapper
obs, reward, done, trunc, info = env.step(0)
print(f"✓ Step works, reward: {reward:.3f}")

env.close()
print("✓ All tests passed!")
