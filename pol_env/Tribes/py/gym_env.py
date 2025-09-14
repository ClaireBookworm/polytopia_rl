import json
import os
from typing import Any, Dict, List, Tuple, Optional, Union

from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway


class TribesGymEnv:
    """Minimal Gym-like API for the Tribes Java engine via Py4J.

    Methods:
      - reset(level_file: str, seed: int, mode: str) -> observation(dict)
      - step(action_index: int) -> (observation, reward, done, info)
      - action_space_n -> int
      - close()
    """

    def __init__(self, classpath_out: str, json_jar: str, port: int = None) -> None:
        # Launch a JVM with the proper classpath if no external gateway is provided.
        if port is None:
            classpath = f"{classpath_out}:{json_jar}"
            # Set working directory to the Tribes directory so Java can find terrainProbs.json
            tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            port = launch_gateway(classpath=classpath, die_on_exit=True, cwd=tribes_dir)
        self._gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))
        self._jvm = self._gateway.jvm
        self._env = self._jvm.core.game.PythonEnv()
        self._last_obs = None

    def reset(self, level_file: str, seed: int = 42, mode: str = "SCORE") -> Dict[str, Any]:
        game_mode = getattr(self._jvm.core.Types.GAME_MODE, mode)
        self._env.initFromLevel(level_file, int(seed), game_mode)
        self._last_obs = json.loads(self._env.observationJson())
        return self._last_obs

    @property
    def action_space_n(self) -> int:
        return int(self._env.actionCount())

    def step(self, action_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Note: Removed setActiveTribeID call due to Py4J method signature issues
        # The game engine should handle turn management automatically
        
        # Get scores before action

        prev_scores = list(self._env.getScores())
        prev_tribe0_score = prev_scores[0]
        
        self._env.stepByIndex(int(action_index))
        obs = json.loads(self._env.observationJson())
        done = bool(self._env.isDone())

        # Get scores after action
        scores = list(self._env.getScores())
        tribe0_score = scores[0]
        other_scores = scores[1:]
        
        # Print only when scores change to reduce spam
        #if prev_tribe0_score != tribe0_score:
        #    print(f"SCORE CHANGE! {prev_tribe0_score} -> {tribe0_score}, others: {other_scores}")
            # print(self.list_actions())
            # print(f"Action: {self.list_actions()[action_index]}")
        
        # Enhanced reward function
        # 1. Progress reward: +1 for each point gained
        progress_reward = (tribe0_score - prev_tribe0_score) / 100.0
        
        # 2. Relative position reward: how well we're doing vs others
        avg_other_score = sum(other_scores) / len(other_scores) if other_scores else 0
        relative_reward = (tribe0_score - avg_other_score) / 1000.0  # Smaller weight
        
        # 3. Small step penalty to encourage ending games faster
        step_penalty = -0.01
        
        # Combine rewards
        reward = progress_reward + relative_reward + step_penalty
        #print(reward)
        
        # Bonus for winning/losing states
        if done:
            if tribe0_score == max(scores):
                reward += 100.0  # Win bonus
                print(f"🏆 WIN! Final score: {tribe0_score} vs {other_scores}")
            else:
                reward -= 100.0  # Loss penalty
                print(f"💀 LOSS! Final score: {tribe0_score} vs {other_scores}")

        info = {
            "tick": int(self._env.getTick()),
            "activeTribeID": int(self._env.getActiveTribeID()),
            "scores": list(scores),
            "score": tribe0_score,  # Add individual score for easier access
            "progress_reward": progress_reward,
            "relative_reward": relative_reward,
        }
        self._last_obs = obs
        return obs, reward, done, info

    def list_actions(self) -> List[dict]:
        return [json.loads(s) for s in list(self._env.listActionsJson())]

    def render(self, mode: str = "ansi") -> Optional[Union[str, "Image.Image"]]:
        """Render the current state.

        - mode="human": print a compact textual view and return None
        - mode="ansi": return a string with the textual view
        - mode="rgb_image": return a Pillow Image (H, W, 3)
        - mode="java": open/update the Java Swing GUI viewer
        """
        if self._last_obs is None:
            self._last_obs = json.loads(self._env.observationJson())
        obs = self._last_obs

        board = obs.get("board", {})
        terrain = board.get("terrain")
        unit_id = board.get("unitID")
        city_id = board.get("cityID")
        resource = board.get("resource")
        size = len(terrain) if terrain else 0

        # Terrain key mapping (must match core.Types.TERRAIN keys)
        terr_to_char = {
            0: ".",  # PLAIN
            1: "s",  # SHALLOW_WATER
            2: "d",  # DEEP_WATER
            3: "m",  # MOUNTAIN
            4: "v",  # VILLAGE
            5: "c",  # CITY
            6: "f",  # FOREST
        }
        
        # Resource symbol mapping (must match core.Types.RESOURCE keys)
        resource_to_char = {
            1: "f",  # FRUIT on PLAIN
            2: "a",  # ANIMAL ("game") on FOREST
            3: "w",  # WHALES on DEEP_WATER
            5: "o",  # ORE on MOUNTAIN
            6: "c",  # CROPS on PLAIN
            7: "r",  # RUINS on any non-blocked tile
            -1: "",  # NONE
        }

        # Resource icon paths and cache
        ICON_FILES = {
            0: "img/resource/fish.png",
            1: "img/resource/fruit.png",
            2: "img/resource/animal.png",
            3: "img/resource/whale.png",
            5: "img/resource/ore.png",
            6: "img/resource/crops.png",
            7: "img/resource/ruins.png",
        }

        icon_cache = {}  # keys: ("raw", key) for original; (key, scale) for resized
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Build text grid
        try:
            scores_py = list(self._env.getScores())
        except Exception:
            scores_py = []
        header = f"tick={obs.get('tick', 0)} activeTribeID={obs.get('activeTribeID', -1)} scores={scores_py} actions={self._env.actionCount()}"
        # Include a few action reprs
        acts = self.list_actions()
        show = ", ".join(a.get("repr", a.get("type", "?")) for a in acts[:6])
        lines: List[str] = []
        lines.append(header)
        lines.append(f"sample_actions=[{show}{', ...' if len(acts)>6 else ''}]")

        if size:
            for i in range(size):
                row_chars = []
                ti = terrain[i]
                ui = unit_id[i]
                ci = city_id[i]
                for j in range(size):
                    ch = terr_to_char.get(int(ti[j]), ".")
                    has_city = int(ci[j]) != -1
                    has_unit = int(ui[j]) != 0
                    if has_city and has_unit:
                        ch = "X"
                    elif has_city:
                        ch = "C"
                    elif has_unit:
                        ch = "U"
                    row_chars.append(ch)
                lines.append("".join(row_chars))

        text = "\n".join(lines)
        if mode == "human":
            print(text)
            return None
        if mode == "ansi":
            return text

        if mode == "java":
            # Leverage the Java GUI components to display the GameState
            self._env.openGui()
            self._env.renderGui()
            return None


        if mode == "rgb_image" or mode == "rgb_array":
            try:
                from PIL import Image, ImageDraw  # type: ignore
            except Exception:
                raise RuntimeError("Pillow (PIL) is required for rgb_image rendering. Install pillow.")

            if not size:
                return None

            # Simple color map per terrain (R,G,B)
            colors = {
                0: (210, 200, 180),  # plain
                1: (100, 170, 230),  # shallow water
                2: (60, 120, 200),   # deep water
                3: (120, 120, 120),  # mountain
                4: (230, 200, 80),   # village
                5: (200, 80, 80),    # city
                6: (60, 140, 80),    # forest
            }
            scale = max(8, 512 // max(1, size))
            img = Image.new("RGB", (size * scale, size * scale), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            for i in range(size):
                for j in range(size):
                    col = colors.get(int(terrain[i][j]), (0, 0, 0))
                    x0, y0 = j * scale, i * scale
                    x1, y1 = x0 + scale, y0 + scale
                    draw.rectangle([x0, y0, x1, y1], fill=col)
                    
                    # Add resource icon if available and tile not covered by city/unit
                    if resource and i < len(resource) and j < len(resource[i]):
                        rkey = int(resource[i][j])
                        icon_rel = ICON_FILES.get(rkey)
                        has_city = int(city_id[i][j]) != -1
                        has_unit = int(unit_id[i][j]) != 0
                        if icon_rel and not (has_city or has_unit):
                            try:
                                from PIL import Image
                                raw = icon_cache.get(("raw", rkey))
                                if raw is None:
                                    raw = Image.open(os.path.join(base_dir, icon_rel)).convert("RGBA")
                                    icon_cache[("raw", rkey)] = raw
                                # resize per scale with margin
                                margin = max(2, scale // 6)
                                size_px = max(6, scale - 2 * margin)
                                icon = icon_cache.get((rkey, size_px))
                                if icon is None:
                                    icon = raw.resize((size_px, size_px), Image.LANCZOS)
                                    icon_cache[(rkey, size_px)] = icon
                                px = x0 + (scale - icon.width) // 2
                                py = y0 + (scale - icon.height) // 2
                                img.paste(icon, (px, py), icon)
                            except Exception:
                                pass  # fall back silently if icon load fails
                    
                    # overlays for city/unit
                    # if int(city_id[i][j]) != -1:
                    #     draw.rectangle([x0+scale//4, y0+scale//4, x1-scale//4, y1-scale//4], outline=(255, 255, 255), width=max(1, scale//8))
                    uid = int(unit_id[i][j])
                    if uid != 0:
                        # small black dot for unit
                        r = max(2, scale//6)
                        cx, cy = x0 + scale//2, y0 + scale//2
                        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 0, 0))
            if mode == "rgb_image":
                return img
            else:
                return img.getdata()

        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        try:
            self._gateway.shutdown()
        except Exception:
            pass


def make_default_env() -> TribesGymEnv:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "out")
    json_jar = os.path.join(base_dir, "lib", "json.jar")
    return TribesGymEnv(classpath_out=out_dir, json_jar=json_jar)


