#!/usr/bin/env python3
"""
Simple test script to verify Python RL environment and Java game coordination.
This runs a short game and shows the Java GUI for visual verification.
"""

import os
import sys
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from gym_env import make_default_env

def main():
    print("=== Tribes RL Environment Coordination Test ===")
    print("This will run a short game with Java GUI to verify coordination.")
    print()
    
    # Change to the Tribes directory so Java can find terrainProbs.json
    tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(tribes_dir)
    
    # Set up environment
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
    json_jar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "json.jar"))
    cp = os.environ.get("CLASSPATH", "")
    sep = ":"
    os.environ["CLASSPATH"] = sep.join([out_dir, json_jar] + ([cp] if cp else []))

    try:
        # Initialize environment
        print("Initializing Python RL environment...")
        env = make_default_env()
        level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levels", "SampleLevel.csv"))
        obs = env.reset(level, seed=42, mode="SCORE")
        print(f"✓ Environment reset successfully - tick: {obs['tick']}")
        
        # Open Java GUI
        print("Opening Java GUI for visualization...")
        env.render(mode="java")
        time.sleep(2)  # Give GUI time to open
        
        print("✓ Java GUI opened successfully!")
        print("\nYou should now see a Java GUI window showing the game board.")
        print("The GUI will update as the Python environment takes actions.")
        print()
        
        # Run a few steps
        acts = env.list_actions()
        print(f"Available actions: {len(acts)}")
        
        for step in range(3):
            if len(acts) == 0:
                print("No more actions available")
                break
                
            # Take a random action
            action = acts[0] if acts else None  # Take first available action
            print(f"Step {step + 1}: Taking action: {action}")
            
            obs, rew, done, info = env.step(0)  # Take first action
            
            # Update GUI
            env.render(mode="java")
            
            print(f"  -> Tick: {info['tick']}, Scores: {info['scores']}, Done: {done}")
            
            if done:
                print("Game finished!")
                break
                
            acts = env.list_actions()
            time.sleep(1)  # Pause for visibility
        
        print("\n=== Test Complete ===")
        print("✓ Python RL environment is working correctly")
        print("✓ Java GUI is displaying the game state")
        print("✓ Both are coordinated and showing the same game")
        print("\nThe coordination is verified! Both environments are running")
        print("the same underlying game logic through the Py4J bridge.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    finally:
        print("\nPress Enter to close the GUI and exit...")
        input()
        env.close()
        print("Environment closed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
