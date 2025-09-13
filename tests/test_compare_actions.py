#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pol_env.Tribes.py.register_env import TribesGymWrapper

def compare_actions():
    """Test multiple actions and see immediate score changes"""
    
    for action_idx in range(3):
        print(f"\n{'='*50}")
        print(f"TESTING ACTION {action_idx}")
        print(f"{'='*50}")
        
        env = TribesGymWrapper()
        obs, info = env.reset()
        
        initial_scores = info.get('scores', [0,0,0,0])
        print(f"Initial scores: {initial_scores}")
        
        # Get action description
        actions = env.tribes_env.list_actions()
        if action_idx < len(actions):
            action_desc = actions[action_idx].get('repr', actions[action_idx].get('type', 'unknown'))
            print(f"Action: {action_desc}")
        
        # Take step
        obs, reward, done, trunc, info = env.step(action_idx)
        
        new_scores = info.get('scores', [0,0,0,0])
        score_change = new_scores[0] - initial_scores[0]
        
        print(f"New scores: {new_scores}")
        print(f"Tribe 0 score change: {score_change}")
        print(f"Reward: {reward:.3f}")
        
        # Save image for this action
        try:
            img = env.tribes_env.render("rgb_image")
            if img:
                img.save(f"action_{action_idx}_result.png")
                print(f"✓ Saved action_{action_idx}_result.png")
        except Exception as e:
            print(f"✗ Image save failed: {e}")
        
        env.close()

if __name__ == "__main__":
    compare_actions()
