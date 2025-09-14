#!/usr/bin/env python3

import sys
import os

# Add repo root so pol_env can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pol_env.Tribes.py.gym_env import make_default_env

def inspect_actions():
    """Inspect the structure of actions in the Polytopia environment"""
    env = make_default_env()
    
    # Initialize environment
    obs = env.reset("levels/SampleLevel.csv", seed=42)
    print(f"Environment initialized with {env.action_space_n} actions")
    
    # Get detailed action information
    actions = env.list_actions()
    print(f"\nDetailed action structure (first 10 actions):")
    
    for i, action in enumerate(actions[:10]):
        print(f"Action {i}: {action}")
    
    # Look for different action types
    action_types = {}
    for action in actions:
        action_type = action.get('type', 'UNKNOWN')
        if action_type not in action_types:
            action_types[action_type] = []
        action_types[action_type].append(action)
    
    print(f"\nAction types found:")
    for action_type, action_list in action_types.items():
        print(f"  {action_type}: {len(action_list)} actions")
        if action_list:
            print(f"    Example: {action_list[0]}")
    
    env.close()

if __name__ == "__main__":
    inspect_actions()
