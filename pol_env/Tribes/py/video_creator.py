import os
import json
import time
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from datetime import datetime

from gym_env import make_default_env


class GameVideoCreator:
    def __init__(self):
        self.env = None
        self.move_history = []
        self.video_frames = []
        
    def setup_environment(self):
        """Setup the game environment"""
        # Change to the Tribes directory so Java can find terrainProbs.json
        tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(tribes_dir)
        
        # Ensure JVM sees classes: prepend out and json.jar to CLASSPATH
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
        json_jar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "json.jar"))
        cp = os.environ.get("CLASSPATH", "")
        sep = ":"
        os.environ["CLASSPATH"] = sep.join([out_dir, json_jar] + ([cp] if cp else []))
        
        self.env = make_default_env()
        level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levels", "SampleLevel.csv"))
        obs = self.env.reset(level, seed=42, mode="SCORE")
        
        self.move_history = []
        self.video_frames = []
        
    def create_video(self, num_steps=20, output_path="game_video.mp4", fps=2):
        """Create a video of the game with move annotations"""
        print("Setting up environment...")
        self.setup_environment()
        
        print(f"Creating video with {num_steps} steps...")
        
        for step in range(num_steps):
            print(f"Processing step {step + 1}/{num_steps}")
            
            # Get available actions
            acts = self.env.list_actions()
            if len(acts) == 0:
                print("No actions available, ending game")
                break
            
            # Choose a random action
            action_idx = random.randint(0, len(acts) - 1)
            action = acts[action_idx]
            
            # Execute the action
            obs, rew, done, info = self.env.step(action_idx)
            
            # Record the move
            player = f"Player {info.get('activeTribeID', 0)}"
            move_desc = action['repr']
            self.move_history.append({
                'step': step,
                'player': player,
                'move': move_desc,
                'scores': info.get('scores', []),
                'reward': rew
            })
            
            # Get game image
            game_img = self.env.render(mode="rgb_image")
            if game_img is not None:
                # Create annotated frame
                annotated_frame = self.create_annotated_frame(game_img, step, info, move_desc, player)
                self.video_frames.append(annotated_frame)
            
            if done:
                print("Game completed!")
                break
        
        # Save video
        print(f"Saving video to {output_path}...")
        self.save_video(output_path, fps)
        
        # Save move history
        history_path = output_path.replace('.mp4', '_moves.json')
        with open(history_path, 'w') as f:
            json.dump(self.move_history, f, indent=2)
        print(f"Move history saved to {history_path}")
        
        self.env.close()
        print("Video creation complete!")
    
    def create_annotated_frame(self, game_img, step, info, move_desc, player):
        """Create a frame with game image and move annotations"""
        # Resize game image
        game_img = game_img.resize((800, 600), Image.Resampling.LANCZOS)
        
        # Create a larger canvas for annotations
        canvas_width = 1200
        canvas_height = 600
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Paste game image on the left
        canvas.paste(game_img, (0, 0))
        
        # Create annotation area on the right
        draw = ImageDraw.Draw(canvas)
        
        try:
            # Try to use a nice font, fallback to default if not available
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw annotations
        x_offset = 820
        y_offset = 20
        
        # Step info
        draw.text((x_offset, y_offset), f"Step {step}", fill='black', font=font_large)
        y_offset += 40
        
        # Current player
        draw.text((x_offset, y_offset), f"Active Player: {player}", fill='blue', font=font_medium)
        y_offset += 30
        
        # Scores
        scores = info.get('scores', [])
        draw.text((x_offset, y_offset), f"Scores: {scores}", fill='green', font=font_medium)
        y_offset += 30
        
        # Current move
        draw.text((x_offset, y_offset), "Last Move:", fill='black', font=font_medium)
        y_offset += 25
        
        # Wrap long move descriptions
        move_lines = self.wrap_text(move_desc, 35)
        for line in move_lines:
            draw.text((x_offset, y_offset), line, fill='red', font=font_small)
            y_offset += 20
        
        y_offset += 20
        
        # Recent moves (last 5)
        draw.text((x_offset, y_offset), "Recent Moves:", fill='black', font=font_medium)
        y_offset += 25
        
        recent_moves = self.move_history[-5:] if len(self.move_history) > 5 else self.move_history
        for move_info in recent_moves:
            move_text = f"Step {move_info['step']}: {move_info['player']}"
            draw.text((x_offset, y_offset), move_text, fill='gray', font=font_small)
            y_offset += 15
            
            move_desc_short = move_info['move'][:40] + "..." if len(move_info['move']) > 40 else move_info['move']
            draw.text((x_offset, y_offset), f"  {move_desc_short}", fill='gray', font=font_small)
            y_offset += 20
        
        return canvas
    
    def wrap_text(self, text, max_length):
        """Wrap text to fit within max_length characters per line"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def save_video(self, output_path, fps):
        """Save the video frames as an MP4 file"""
        if not self.video_frames:
            print("No frames to save!")
            return
        
        # Get frame dimensions
        height, width = self.video_frames[0].size[1], self.video_frames[0].size[0]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in self.video_frames:
            # Convert PIL image to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_cv)
        
        video_writer.release()
        print(f"Video saved with {len(self.video_frames)} frames")


def main():
    creator = GameVideoCreator()
    
    # Create a video with 30 steps
    creator.create_video(
        num_steps=30,
        output_path="polytopia_game.mp4",
        fps=1  # 1 frame per second
    )


if __name__ == "__main__":
    main()
