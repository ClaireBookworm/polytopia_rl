import os
import json
import time
import random


from gym_env import make_default_env


def main():
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
    steps = 5
    for t in range(steps):
        if len(acts) == 0:
            print("no_actions")
            break 
        obs, rew, done, info = env.step(0)
        img = env.render(mode="rgb_image")
        if img is not None:
<<<<<<< HEAD
            # Save image to the Tribes directory (same level as this script's parent)
            img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"img_step_{t}.png")
            img.save(img_path)
=======
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            img.save(os.path.join(out_dir, f"img_step_{t}.png"))
>>>>>>> 2cbc661b419601027cf8b04c9a57c60f6357a976
        print("t", t, "tick", info["tick"], "scores", info["scores"], "done", done)
        if done:
            break
        acts = env.list_actions()
        action = random.randint(0, len(acts) - 1)
        print('taking action:', acts[action])
        obs, rew, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()


