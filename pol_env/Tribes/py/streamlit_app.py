import os
import json
import time
import random
import threading
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image
import gymnasium as gym
import imageio.v2 as imageio
import streamlit as st

from gym_env import make_default_env

# Configure Streamlit page
st.set_page_config(
    page_title="Polytopia Game Visualizer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'env': None,
        'current_step': 0,
        'move_history': [],
        'is_running': False,
        'auto_steps_remaining': 0,
        'speed': 1.0,
        'last_image': None,
        'last_info': None,
        'setup_error': None,
        'recording': False,
        'video_frames': [],
        'video_filename': None,
        'frames_dir': None,
        'video_dir': None,
        'game_completed': False,
        'manual_play_stopped': False,
        'run_name': None
    }

def start_game():
    """Initialize a new game"""
    try:
        st.info("Setting up game environment...")
        
        # Setup environment
        tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        original_cwd = os.getcwd()
        os.chdir(tribes_dir)
        
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
        json_jar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "json.jar"))
        cp = os.environ.get("CLASSPATH", "")
        sep = ":"
        os.environ["CLASSPATH"] = sep.join([out_dir, json_jar] + ([cp] if cp else []))
        
        # Create environment
        st.session_state.game_state['env'] = make_default_env()
        
        # Create run name for video recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.game_state['run_name'] = f"polytopia_game_{timestamp}"
        
        # Load level
        level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levels", "SampleLevel.csv"))
        obs = st.session_state.game_state['env'].reset(level, seed=42, mode="SCORE")
        
        # Reset game state
        st.session_state.game_state['current_step'] = 0
        st.session_state.game_state['move_history'] = []
        st.session_state.game_state['last_image'] = None
        st.session_state.game_state['last_info'] = None
        st.session_state.game_state['setup_error'] = None
        st.session_state.game_state['game_completed'] = False
        st.session_state.game_state['manual_play_stopped'] = False
        
        # Initialize video recording
        start_recording()
        
        os.chdir(original_cwd)
        st.success("Game setup complete!")
        return True
        
    except Exception as e:
        st.error(f"Error setting up game: {e}")
        st.session_state.game_state['setup_error'] = str(e)
        os.chdir(original_cwd)
        return False

def start_recording():
    """Start video recording"""
    try:
        # Create frames and videos directories if they don't exist
        st.session_state.game_state['frames_dir'] = f"frames/{st.session_state.game_state['run_name']}"
        st.session_state.game_state['video_dir'] = "videos"
        os.makedirs(st.session_state.game_state['frames_dir'], exist_ok=True)
        os.makedirs(st.session_state.game_state['video_dir'], exist_ok=True)
        
        # Initialize video recording
        st.session_state.game_state['video_frames'] = []
        st.session_state.game_state['recording'] = True
        st.session_state.game_state['video_filename'] = f"{st.session_state.game_state['video_dir']}/{st.session_state.game_state['run_name']}.mp4"
        
    except Exception as e:
        st.error(f"Error starting recording: {e}")

def record_frame(game_img):
    """Record a frame for the video"""
    try:
        if st.session_state.game_state['recording'] and game_img:
            # Save frame as image in frames directory
            frame_filename = f"{st.session_state.game_state['frames_dir']}/frame_{st.session_state.game_state['current_step']:04d}.png"
            game_img.save(frame_filename)
            st.session_state.game_state['video_frames'].append(frame_filename)
    except Exception as e:
        st.error(f"Error recording frame: {e}")

def finish_recording():
    """Finish video recording and save to file"""
    try:
        if not st.session_state.game_state['video_frames']:
            st.warning("No frames to save")
            return
        
        # Create video from frames using imageio
        with imageio.get_writer(st.session_state.game_state['video_filename'], fps=2) as writer:
            for frame_path in st.session_state.game_state['video_frames']:
                if os.path.exists(frame_path):
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)
        
        st.success(f"Video created: {st.session_state.game_state['video_filename']}")
        
    except Exception as e:
        st.error(f"Error finishing recording: {e}")

def next_step():
    """Execute next game step"""
    try:
        if not st.session_state.game_state['env']:
            st.error("Game not started")
            return False
        
        # Check if manual play is stopped
        if st.session_state.game_state['manual_play_stopped']:
            st.warning("Game is stopped - cannot take more steps")
            return False
        
        acts = st.session_state.game_state['env'].list_actions()
        if len(acts) == 0:
            st.warning("No actions available")
            return False
        
        # Choose random action
        action_idx = random.randint(0, len(acts) - 1)
        action = acts[action_idx]
        
        # Execute action
        obs, rew, done, info = st.session_state.game_state['env'].step(action_idx)
        
        # Record move
        player = f"Player {info.get('activeTribeID', 0)}"
        move_desc = action['repr']
        st.session_state.game_state['move_history'].append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'player': player,
            'move': move_desc,
            'step': st.session_state.game_state['current_step']
        })
        
        # Update game state
        st.session_state.game_state['current_step'] += 1
        st.session_state.game_state['last_info'] = info
        
        # Get game image
        try:
            game_img = st.session_state.game_state['env'].render(mode="rgb_image")
            if game_img:
                st.session_state.game_state['last_image'] = game_img
                
                # Record frame for video
                if st.session_state.game_state['recording']:
                    record_frame(game_img)
        except Exception as e:
            st.error(f"Error rendering image: {e}")
        
        # Check if game is completed
        if done:
            st.session_state.game_state['game_completed'] = True
            if st.session_state.game_state['recording']:
                finish_recording()
            st.success("🎉 Game completed!")
        
        return True
        
    except Exception as e:
        st.error(f"Error in next_step: {e}")
        return False

def save_move_history():
    """Save move history to JSON file"""
    if not st.session_state.game_state['move_history']:
        st.warning("No moves to save")
        return
    
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_moves_{timestamp}.json"
        
        # Prepare data for JSON
        game_data = {
            'game_info': {
                'total_steps': st.session_state.game_state['current_step'],
                'total_moves': len(st.session_state.game_state['move_history']),
                'created_at': datetime.now().isoformat(),
                'final_scores': st.session_state.game_state['last_info'].get('scores', []) if st.session_state.game_state['last_info'] else []
            },
            'moves': st.session_state.game_state['move_history']
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        st.success(f"Move history saved to {filename}")
        
    except Exception as e:
        st.error(f"Error saving move history: {e}")

def stop_game():
    """Stop the current game"""
    st.session_state.game_state['is_running'] = False
    st.session_state.game_state['auto_steps_remaining'] = 0
    st.session_state.game_state['manual_play_stopped'] = True
    
    # Finish recording if it was active
    if st.session_state.game_state['recording']:
        finish_recording()
        st.session_state.game_state['recording'] = False
    
    save_move_history()
    st.info("Game stopped - Move history saved")

# Main Streamlit UI
st.title("🎮 Polytopia Game Visualizer")

# Sidebar controls
st.sidebar.header("Game Controls")

# Game status
if st.session_state.game_state['env']:
    st.sidebar.success("✅ Game Ready")
    st.sidebar.metric("Current Step", st.session_state.game_state['current_step'])
    if st.session_state.game_state['last_info']:
        st.sidebar.metric("Active Player", st.session_state.game_state['last_info'].get('activeTribeID', 'Unknown'))
else:
    st.sidebar.warning("⚠️ Game Not Started")

# Control buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🎮 Start New Game", disabled=st.session_state.game_state['is_running']):
        start_game()
        st.rerun()

with col2:
    if st.button("⏹️ Stop Game", disabled=not st.session_state.game_state['env'] or st.session_state.game_state['manual_play_stopped']):
        stop_game()
        st.rerun()

# Step controls
st.sidebar.header("Step Controls")

if st.sidebar.button("▶️ Next Step", disabled=not st.session_state.game_state['env'] or st.session_state.game_state['manual_play_stopped']):
    next_step()
    st.rerun()

# Auto-play controls
st.sidebar.header("Auto-Play")

auto_steps = st.sidebar.number_input("Number of steps", min_value=1, max_value=100, value=10)
speed = st.sidebar.slider("Speed (seconds per step)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

if st.sidebar.button("🚀 Run Auto-Play", disabled=not st.session_state.game_state['env'] or st.session_state.game_state['is_running']):
    st.session_state.game_state['is_running'] = True
    st.session_state.game_state['auto_steps_remaining'] = auto_steps
    st.session_state.game_state['speed'] = speed
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i in range(auto_steps):
        if not st.session_state.game_state['is_running']:
            break
            
        if next_step():
            progress_bar.progress((i + 1) / auto_steps)
            status_text.text(f"Step {i + 1}/{auto_steps}")
            time.sleep(speed)
        else:
            break
    
    st.session_state.game_state['is_running'] = False
    progress_bar.empty()
    status_text.empty()
    st.rerun()

# Download controls
st.sidebar.header("Downloads")

if st.sidebar.button("💾 Save History", disabled=not st.session_state.game_state['move_history']):
    save_move_history()

if st.sidebar.button("📹 Download Video", disabled=not st.session_state.game_state['video_filename'] or not os.path.exists(st.session_state.game_state['video_filename'])):
    if st.session_state.game_state['video_filename'] and os.path.exists(st.session_state.game_state['video_filename']):
        with open(st.session_state.game_state['video_filename'], 'rb') as f:
            video_data = f.read()
        st.download_button(
            label="Download Video",
            data=video_data,
            file_name=os.path.basename(st.session_state.game_state['video_filename']),
            mime="video/mp4"
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Game Visualization")
    
    if st.session_state.game_state['last_image']:
        st.image(st.session_state.game_state['last_image'], caption=f"Step {st.session_state.game_state['current_step']}", use_column_width=True)
    else:
        st.info("Click 'Start New Game' to begin...")

with col2:
    st.header("Game Information")
    
    if st.session_state.game_state['last_info']:
        info = st.session_state.game_state['last_info']
        st.json({
            'Current Step': st.session_state.game_state['current_step'],
            'Active Player': info.get('activeTribeID', 'Unknown'),
            'Scores': info.get('scores', []),
            'Done': info.get('done', False),
            'Tick': info.get('tick', 'Unknown'),
            'Available Actions': len(st.session_state.game_state['env'].list_actions()) if st.session_state.game_state['env'] else 0
        })
    else:
        st.info("Game not started")

# Move history
st.header("Move History")

if st.session_state.game_state['move_history']:
    # Show last 10 moves
    recent_moves = st.session_state.game_state['move_history'][-10:]
    for move in reversed(recent_moves):
        st.text(f"[{move['timestamp']}] {move['player']}: {move['move']}")
else:
    st.info("No moves yet")

# Footer
st.markdown("---")
st.markdown("**Polytopia Game Visualizer** - Built with Streamlit")
