## Polytopia RL

### Project Summary
Polytopia RL is a research framework for training and analyzing reinforcement learning agents in a turn-based strategy environment. We combine the Tribes Java engine (forward model, rules, GUI) with a lightweight Python bridge (Py4J) and Gym-style APIs for rapid prototyping, evaluation, and visualization.

Key features:
- Java engine (Tribes) for deterministic forward modeling and rich game dynamics
- Python bridge exposing reset/step/list_actions/observation JSON
- Multiple renderers: Java GUI, ANSI text, PIL images, and a simple web GUI
- GitHub Codespaces support for one‑click cloud dev environments

### Demo (Video)
Watch a short demo here: https://www.youtube.com/watch?v=xPcRPhiPnsg

### Project Slideshow
Slides with methods and visuals: https://docs.google.com/presentation/d/1oripsNAIIUnl0mX7U4JyGPdtTLTwYCnO9eWK9A8KGiU/edit?usp=sharing

### Simple Web GUI
- Location: `pol_env/Tribes/py/simple_web_gui.py`
- Usage (local):
  1) Create and activate a venv
     - `python3 -m venv pol_env/Tribes/.venv`
     - `source pol_env/Tribes/.venv/bin/activate`
  2) Install requirements
     - `pip install -U pip wheel`
     - `pip install -r pol_env/Tribes/py/requirements.txt`
     - For the web GUI: `pip install gymnasium imageio numpy`
  3) Run the server
     - `cd pol_env/Tribes/py`
     - `python simple_web_gui.py`
     - Open `http://localhost:8000`

### Java UI and Python Bridge (quickstart)
- Java UI (after compiling):
  - `cd pol_env/Tribes`
  - `java -cp out:lib/json.jar Play`
- Python demo:
  - `cd pol_env/Tribes/py`
  - `python run_gym.py`

### Codespaces
This repo includes `.devcontainer/devcontainer.json` so you can launch a cloud dev environment from the GitHub UI (Code → Codespaces → Create codespace on main). It installs Python deps, Java, and compiles the engine automatically.
