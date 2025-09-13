#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pol_env.Tribes.py.register_env import TribesGymWrapper

env = TribesGymWrapper()
obs, info = env.reset()

print("=== GAME STATE ===")
print(env.tribes_env.render("ansi"))

print("\n=== AVAILABLE ACTIONS ===")
actions = env.tribes_env.list_actions()
for i, action in enumerate(actions[:5]):
    print(f"{i}: {action.get('repr', action.get('type', 'unknown'))}")

print(f"\nTotal actions: {len(actions)}")
env.close()
