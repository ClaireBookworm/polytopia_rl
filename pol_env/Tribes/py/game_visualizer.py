import os
import json
import time
import random
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
from datetime import datetime

from gym_env import make_default_env


class GameVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Polytopia Game Visualizer")
        self.root.geometry("1400x800")
        
        # Game state
        self.env = None
        self.current_step = 0
        self.game_images = []
        self.move_history = []
        self.is_running = False
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Game visualization
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Game image display
        self.image_label = ttk.Label(left_frame, text="Game will start here...", 
                                   font=("Arial", 16), anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="Start New Game", 
                                     command=self.start_new_game)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.step_button = ttk.Button(control_frame, text="Next Step", 
                                    command=self.next_step, state=tk.DISABLED)
        self.step_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.auto_button = ttk.Button(control_frame, text="Auto Play", 
                                    command=self.toggle_auto_play)
        self.auto_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_label = ttk.Label(control_frame, text="Speed:")
        speed_label.pack(side=tk.LEFT, padx=(10, 5))
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=3.0, 
                              variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        # Right side - Move history and game info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Game info
        info_frame = ttk.LabelFrame(right_frame, text="Game Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Move history
        history_frame = ttk.LabelFrame(right_frame, text="Move History")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to start")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_environment(self):
        """Setup the game environment"""
        try:
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
            
            self.current_step = 0
            self.move_history = []
            self.game_images = []
            
            return True
        except Exception as e:
            self.status_var.set(f"Error setting up environment: {str(e)}")
            return False
    
    def start_new_game(self):
        """Start a new game"""
        self.status_var.set("Setting up new game...")
        self.start_button.config(state=tk.DISABLED)
        
        # Run setup in a separate thread to avoid blocking GUI
        def setup_thread():
            if self.setup_environment():
                self.root.after(0, self.on_game_ready)
            else:
                self.root.after(0, self.on_setup_failed)
        
        threading.Thread(target=setup_thread, daemon=True).start()
    
    def on_game_ready(self):
        """Called when game setup is complete"""
        self.start_button.config(state=tk.NORMAL)
        self.step_button.config(state=tk.NORMAL)
        self.status_var.set("Game ready! Click 'Next Step' to start playing.")
        
        # Display initial game state
        self.update_display()
    
    def on_setup_failed(self):
        """Called when game setup fails"""
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Failed to setup game. Check console for errors.")
    
    def next_step(self):
        """Execute the next step in the game"""
        if not self.env:
            return
            
        try:
            acts = self.env.list_actions()
            if len(acts) == 0:
                self.add_move_to_history("No actions available", "System")
                self.status_var.set("Game over - no actions available")
                return
            
            # Choose a random action for now (you can modify this to be user input)
            action_idx = random.randint(0, len(acts) - 1)
            action = acts[action_idx]
            
            # Execute the action
            obs, rew, done, info = self.env.step(action_idx)
            
            # Record the move
            player = f"Player {info.get('activeTribeID', 0)}"
            move_desc = f"Step {self.current_step}: {action['repr']}"
            self.add_move_to_history(move_desc, player)
            
            # Update game info
            self.update_game_info(info, rew, done)
            
            # Update display
            self.update_display()
            
            self.current_step += 1
            
            if done:
                self.status_var.set("Game completed!")
                self.step_button.config(state=tk.DISABLED)
            else:
                self.status_var.set(f"Step {self.current_step} completed")
                
        except Exception as e:
            self.status_var.set(f"Error during step: {str(e)}")
    
    def toggle_auto_play(self):
        """Toggle automatic play mode"""
        if not self.is_running:
            self.is_running = True
            self.auto_button.config(text="Stop Auto Play")
            self.step_button.config(state=tk.DISABLED)
            self.start_button.config(state=tk.DISABLED)
            self.auto_play()
        else:
            self.is_running = False
            self.auto_button.config(text="Auto Play")
            self.step_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
    
    def auto_play(self):
        """Automatically play the game"""
        if not self.is_running or not self.env:
            return
            
        self.next_step()
        
        # Schedule next step based on speed setting
        delay = int(1000 / self.speed_var.get())  # Convert to milliseconds
        self.root.after(delay, self.auto_play)
    
    def update_display(self):
        """Update the visual display"""
        if not self.env:
            return
            
        try:
            # Get the current game image
            img = self.env.render(mode="rgb_image")
            if img is not None:
                # Resize image to fit the display
                img = img.resize((600, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
            else:
                # Fallback to text display
                text_state = self.env.render(mode="ansi")
                self.image_label.config(image="", text=text_state)
                
        except Exception as e:
            self.image_label.config(image="", text=f"Error rendering: {str(e)}")
    
    def update_game_info(self, info, reward, done):
        """Update the game information display"""
        info_text = f"""Current Step: {self.current_step}
Active Player: {info.get('activeTribeID', 'Unknown')}
Scores: {info.get('scores', [])}
Reward: {reward}
Done: {done}
Tick: {info.get('tick', 'Unknown')}

Available Actions: {len(self.env.list_actions())}
"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def add_move_to_history(self, move, player):
        """Add a move to the history display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] {player}: {move}\n"
        
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)  # Scroll to bottom
        
        # Also store in our internal history
        self.move_history.append({
            'timestamp': timestamp,
            'player': player,
            'move': move,
            'step': self.current_step
        })


def main():
    root = tk.Tk()
    app = GameVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
