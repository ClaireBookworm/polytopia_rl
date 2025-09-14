#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pol_env.Tribes.py.register_env import TribesGymWrapper

def test_immediate_reward():
    env = TribesGymWrapper()
    obs, info = env.reset()
    
    print("=== INITIAL STATE ===")
    print(f"Initial scores: {info.get('scores', 'N/A')}")
    print(f"Valid actions: {info['valid_actions']}")
    
    # Get available actions
    actions = env.tribes_env.list_actions()
    print(f"\n=== AVAILABLE ACTIONS (showing first 3) ===")
    for i in range(min(3, len(actions))):
        action = actions[i]
        print(f"Action {i}: {action.get('repr', action.get('type', 'unknown'))}")
    
    # Choose action 0, 1, or 2
    chosen_action = 1  # You can change this
    print(f"\n=== CHOOSING ACTION {chosen_action} ===")
    
    # Take the step
    obs, reward, done, trunc, info = env.step(chosen_action)
    
    print(f"Reward: {reward:.3f}")
    print(f"New scores: {info.get('scores', 'N/A')}")
    print(f"Done: {done}")
    
    # Save game state as image
    try:
        img = env.tribes_env.render("rgb_image")
        if img:
            img.save("game_state.png")
            print("✓ Saved game_state.png")
        else:
            print("✗ Could not generate image")
    except Exception as e:
        print(f"✗ Image save failed: {e}")
    
    # Also save text representation
    text_state = env.tribes_env.render("ansi")
    with open("game_state.txt", "w") as f:
        f.write(text_state)
    print("✓ Saved game_state.txt")
    
    env.close()

if __name__ == "__main__":
    test_immediate_reward()
