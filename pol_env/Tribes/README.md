## Quickstart

### 1) Compile Java Code First
Before running the Java environment, you need to compile the source code:

**Prerequisites:**
- Java 8+ with JDK (`java -version` and `javac -version`)
- From repository root:

```bash
cd pol_env/Tribes
mkdir -p out
find src -name "*.java" -exec javac -cp "lib/json.jar" -d out -sourcepath src {} +
```

### 2) Standalone Java Player UI
- After compilation, run the player UI:
```bash
java -cp out:lib/json.jar Play
```
- The UI parameters are defined in `pol_env/Tribes/play.json`.
- **Note:** Requires a display server (GUI). On headless servers, you'll get a `HeadlessException`.

### 3) Python Bridge (Py4J) + Installation
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

## Compiling Java Code

Before running the Python environment, you need to compile the Java source code to create the `out` directory with compiled classes.

### Prerequisites
- Java 8+ (`java -version`)
- The `lib/json.jar` file should be present in the `lib/` directory

### Compilation Steps
From the `pol_env/Tribes` directory:

```bash
# Create the out directory
mkdir -p out

# Compile all Java source files (corrected command)
find src -name "*.java" -exec javac -cp "lib/json.jar" -d out -sourcepath src {} +
```

### Verification
After compilation, you should see:
- An `out/` directory containing compiled `.class` files
- The key file `out/core/game/PythonEnv.class` should exist

### Troubleshooting
- If you get "command not found: javac", make sure Java JDK is installed (not just JRE)
- On macOS: `brew install openjdk` or download from Oracle
- On Ubuntu: `sudo apt install openjdk-11-jdk`
- On Windows: Download JDK from Oracle or use `choco install openjdk`

Once the `out` directory is created with compiled classes, you can proceed with the Python setup and run the environment.
