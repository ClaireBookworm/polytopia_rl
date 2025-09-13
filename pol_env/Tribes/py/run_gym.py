import os
import json
import time
import random


from gym_env import make_default_env


def main():
    # Change to the Tribes directory so Java can find terrainProbs.json
    tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(tribes_dir)
    
    # Ensure JVM sees classes: prepend out and json.jar to CLASSPATH
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
    json_jar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "json.jar"))
    cp = os.environ.get("CLASSPATH", "")
    sep = ":"
    os.environ["CLASSPATH"] = sep.join([out_dir, json_jar] + ([cp] if cp else []))

    env = make_default_env()
    level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levels", "SampleLevel.csv"))
    obs = env.reset(level, seed=42, mode="SCORE")
    print("reset_ok", obs["tick"]) if isinstance(obs, dict) else print("reset_ok")

    acts = env.list_actions()
    print("n_actions", len(acts))
    print(env.render(mode="ansi"))
    
    # Skip Java GUI on headless servers
    print("Running on headless server - using text and image rendering instead")
    
    steps = 5
    for t in range(steps):
        if len(acts) == 0:
            print("no_actions")
            break 
        obs, rew, done, info = env.step(0)
        img = env.render(mode="rgb_image")
        if img is not None:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            img.save(os.path.join(out_dir, f"img_step_{t}.png"))
        print("t", t, "tick", info["tick"], "scores", info["scores"], "done", done)
        
        # Show text state after each step
        print(env.render(mode="ansi"))
        
        if done:
            break
        acts = env.list_actions()
        action = random.randint(0, len(acts) - 1)
        print('taking action:', acts[action])
        obs, rew, done, info = env.step(action)
        
        # Show text state after action
        print(env.render(mode="ansi"))

    env.close()


if __name__ == "__main__":
    main()


