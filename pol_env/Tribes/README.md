## Quickstart

### 1) Standalone Java Player UI
- Prereq: Java 8+ (`java -version`)
- From repository root:
```bash
cd pol_env/Tribes
java -cp out:lib/json.jar Play
```
- The UI parameters are defined in `pol_env/Tribes/play.json`.

### 2) Python Bridge (Py4J) + Installation
- Prereqs: Python 3.9+ and pip
- From repository root:
```bash
python3 -m venv pol_env/Tribes/.venv
source pol_env/Tribes/.venv/bin/activate
pip install -U pip wheel
pip install -r pol_env/Tribes/py/requirements.txt
```
- Or, if you're already in `pol_env/Tribes`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r py/requirements.txt
```
- Run the demo (from repo root):
```bash
cd pol_env/Tribes/py
python run_gym.py
```
- Or, if you're already in `pol_env/Tribes`:
```bash
source .venv/bin/activate
cd py
python run_gym.py
```
- Notes:
  - No manual `CLASSPATH` needed; the Python code launches the JVM with the correct classpath.
  - Images from the demo are saved to `pol_env/Tribes/img_step_*.png`.
