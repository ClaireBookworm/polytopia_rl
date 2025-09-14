#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pol_env.Tribes.py.register_env import TribesGymWrapper

env = TribesGymWrapper()
obs, info = env.reset()

print(f"Obs shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Valid actions: {info['valid_actions']}")

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print(f"Step {i}: action={action}, reward={reward:.3f}, valid={info['valid_actions']}")
    if done:
        print("Episode done!")
        break

env.close()
