#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from pol_env.Tribes.py.register_env import TribesGymWrapper

def test_episode():
    env = TribesGymWrapper()
    obs, info = env.reset()
    
    rewards = []
    scores = []
    
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        
        rewards.append(reward)
        if 'scores' in info:
            scores.append(info['scores'])
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    return rewards, scores

print("Testing episode...")
rewards, scores = test_episode()

print(f"\nRewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}")

if scores:
    tribe0_scores = [s[0] for s in scores]
    print(f"Tribe 0 scores: {tribe0_scores[0]} -> {tribe0_scores[-1]} (change: {tribe0_scores[-1] - tribe0_scores[0]})")
