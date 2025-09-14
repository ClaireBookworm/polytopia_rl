import os
import json
import time
import random
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from urllib.parse import urlparse
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

from gym_env import make_default_env


class GameState:
    def __init__(self):
        self.env = None
        self.current_step = 0
        self.move_history = []
        self.is_running = False
        self.auto_steps_remaining = 0
        self.speed = 1.0
        self.last_image = None
        self.last_info = None
        self.setup_error = None
        self.recording = False
        self.video_frames = []
        self.video_writer = None
        self.video_filename = None
        self.game_completed = False
        self.manual_play_stopped = False


class SimpleWebHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, game_state=None, **kwargs):
        self.game_state = game_state
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_html()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/image':
            self.serve_image()
        elif path == '/api/next_step':
            self.serve_next_step()
        elif path == '/api/auto_play':
            self.serve_auto_play()
        elif path == '/api/stop':
            self.serve_stop()
        elif path == '/api/start_game':
            self.serve_start_game()
        elif path == '/api/save_history':
            self.serve_save_history()
        elif path == '/api/download_video':
            self.serve_download_video()
        elif path == '/api/download_json':
            self.serve_download_json()
        else:
            self.send_error(404)
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/auto_play':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            self.serve_auto_play_post(data)
        else:
            self.send_error(404)
    
    def serve_html(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polytopia Game Visualizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
        }
        .game-area {
            flex: 1;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .info-area {
            width: 400px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .game-image {
            width: 100%;
            max-width: 800px;
            height: 600px;
            border: 2px solid #ddd;
            border-radius: 4px;
            background: #f9f9f9;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: #666;
        }
        .controls {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .control-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        #speedSlider {
            width: 150px;
            margin: 0 10px;
        }
        
        #speedDisplay {
            font-weight: bold;
            color: #2c3e50;
            min-width: 40px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        input[type="number"] {
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 80px;
        }
        input[type="range"] {
            width: 100px;
        }
        .info-panel {
            margin-bottom: 20px;
        }
        .info-panel h3 {
            margin-top: 0;
            color: #333;
        }
        .info-content {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .history {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
        }
        .status {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #333;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .error {
            color: red;
            background: #ffe6e6;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background-color: white;
            margin: 15% auto;
            padding: 20px;
            border-radius: 8px;
            width: 400px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .modal h3 {
            margin-top: 0;
            color: #333;
        }
        .modal-buttons {
            margin-top: 20px;
        }
        .modal-buttons button {
            margin: 0 10px;
            padding: 10px 20px;
        }
        .download-btn {
            background: #28a745;
        }
        .download-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="game-area">
            <h2>Polytopia Game Visualizer</h2>
            <div class="game-image" id="gameImage">
                Click 'Start New Game' to begin...
            </div>
            
            <div class="controls">
                <div class="control-row">
                    <button id="startBtn" onclick="startGame()">Start New Game</button>
                    <button id="stepBtn" onclick="nextStep()" disabled>Next Step</button>
                    <button id="stopBtn" onclick="stopGame()" disabled>Stop Game</button>
                    <button id="saveBtn" onclick="saveHistory()" disabled>Save History</button>
                </div>
                <div class="control-row">
                    <label>Auto-play:</label>
                    <input type="number" id="stepsInput" value="10" min="1" max="1000">
                    <button id="autoBtn" onclick="startAutoPlay()" disabled>Run Steps</button>
                </div>
                <div class="control-row">
                    <label>Speed:</label>
                    <input type="range" id="speedSlider" min="0.1" max="5" step="0.1" value="1" oninput="updateSpeed()">
                    <span id="speedDisplay">1.0s</span>
                </div>
            </div>
        </div>
        
        <div class="info-area">
            <div class="info-panel">
                <h3>Game Information</h3>
                <div class="info-content" id="gameInfo">
                    Game not started
                </div>
            </div>
            
            <div class="info-panel">
                <h3>Move History</h3>
                <div class="history" id="moveHistory">
                    No moves yet
                </div>
            </div>
        </div>
    </div>
    
    <div class="status" id="statusBar">
        Ready to start
    </div>
    
    <!-- Game Complete Modal -->
    <div id="gameCompleteModal" class="modal">
        <div class="modal-content">
            <h3>🎉 Game Complete!</h3>
            <p>Your game has finished! Would you like to download the recording and data?</p>
            <div class="modal-buttons">
                <button class="download-btn" onclick="downloadVideo()">📹 Download Video</button>
                <button class="download-btn" onclick="downloadJSON()">📄 Download JSON</button>
                <button onclick="closeModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        let gameState = {
            isRunning: false,
            autoStepsRemaining: 0,
            currentSpeed: 1.0
        };
        
        function updateSpeed() {
            const speedSlider = document.getElementById('speedSlider');
            const speedDisplay = document.getElementById('speedDisplay');
            const speed = parseFloat(speedSlider.value);
            
            gameState.currentSpeed = speed;
            speedDisplay.textContent = speed.toFixed(1) + 's';
        }

        function updateStatus(message) {
            document.getElementById('statusBar').textContent = message;
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            document.querySelector('.game-area').insertBefore(errorDiv, document.querySelector('.controls'));
            
            // Remove error after 5 seconds
            setTimeout(() => {
                if (errorDiv.parentNode) {
                    errorDiv.parentNode.removeChild(errorDiv);
                }
            }, 5000);
        }

        function updateSpeed(value) {
            document.getElementById('speedValue').textContent = value + 'x';
        }

        function startGame() {
            updateStatus('Starting new game...');
            document.getElementById('startBtn').disabled = true;
            
            fetch('/api/start_game')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus('Game ready! Click Next Step to start playing.');
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stepBtn').disabled = false;
                        document.getElementById('autoBtn').disabled = false;
                        document.getElementById('saveBtn').disabled = false;
                        updateDisplay();
                    } else {
                        updateStatus('Failed to start game');
                        document.getElementById('startBtn').disabled = false;
                        showError('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    updateStatus('Failed to start game');
                    document.getElementById('startBtn').disabled = false;
                    showError('Network error: ' + error.message);
                });
        }

        function nextStep() {
            fetch('/api/next_step')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateDisplay();
                        updateStatus(`Step ${data.step} completed`);
                        
                        // Check if game is completed
                        if (data.game_completed) {
                            showGameCompleteModal();
                        }
                    } else {
                        if (data.error === 'Game stopped by user') {
                            updateStatus('Game stopped by user');
                            document.getElementById('stepBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = true;
                        } else {
                            updateStatus('Error: ' + data.error);
                            showError('Step error: ' + data.error);
                        }
                    }
                })
                .catch(error => {
                    updateStatus('Error during step');
                    showError('Network error: ' + error.message);
                });
        }

        function startAutoPlay() {
            const steps = parseInt(document.getElementById('stepsInput').value);
            if (steps <= 0) {
                alert('Please enter a positive number of steps');
                return;
            }
            
            gameState.isRunning = true;
            gameState.autoStepsRemaining = steps;
            
            document.getElementById('autoBtn').disabled = true;
            document.getElementById('stepBtn').disabled = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            updateStatus(`Auto-playing ${steps} steps...`);
            
            fetch('/api/auto_play', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({steps: steps, speed: gameState.currentSpeed})
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    showError('Auto-play error: ' + data.error);
                    stopGame();
                }
            })
                .catch(error => {
                    showError('Network error: ' + error.message);
                    stopGame();
                });
        }

        function stopGame() {
            gameState.isRunning = false;
            gameState.autoStepsRemaining = 0;
            
            document.getElementById('autoBtn').disabled = false;
            document.getElementById('stepBtn').disabled = true;  // Disable step button when stopped
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            updateStatus('Stopping game...');
            
            fetch('/api/stop')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus('Game stopped - Move history saved');
                    }
                });
        }

        function saveHistory() {
            fetch('/api/save_history')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus('Move history saved to JSON file!');
                    } else {
                        updateStatus('Failed to save history');
                        showError('Save error: ' + data.error);
                    }
                })
                .catch(error => {
                    updateStatus('Failed to save history');
                    showError('Network error: ' + error.message);
                });
        }

        function updateDisplay() {
            // Update game image
            const gameImage = document.getElementById('gameImage');
            const img = new Image();
            img.onload = function() {
                gameImage.innerHTML = '';
                gameImage.appendChild(img);
            };
            img.onerror = function() {
                gameImage.innerHTML = 'Image not available';
            };
            img.src = '/api/image?' + new Date().getTime(); // Cache busting
            
            // Update game info
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('gameInfo').textContent = data.info;
                    document.getElementById('moveHistory').innerHTML = data.history;
                })
                .catch(error => {
                    console.error('Error updating display:', error);
                });
        }

        // Auto-refresh display every 500ms when auto-playing
        setInterval(() => {
            if (gameState.isRunning) {
                updateDisplay();
            }
        }, 500);

        function showGameCompleteModal() {
            document.getElementById('gameCompleteModal').style.display = 'block';
        }

        function closeModal() {
            document.getElementById('gameCompleteModal').style.display = 'none';
        }

        function downloadVideo() {
            window.open('/api/download_video', '_blank');
            updateStatus('Video download started!');
        }

        function downloadJSON() {
            window.open('/api/download_json', '_blank');
            updateStatus('JSON download started!');
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('gameCompleteModal');
            if (event.target == modal) {
                closeModal();
            }
        }

        // Initial display update
        updateDisplay();
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_status(self):
        info_text = "Game not started"
        history_html = "No moves yet"
        
        if self.game_state.env:
            if self.game_state.last_info:
                info_text = f"""Current Step: {self.game_state.current_step}
Active Player: {self.game_state.last_info.get('activeTribeID', 'Unknown')}
Scores: {self.game_state.last_info.get('scores', [])}
Done: {self.game_state.last_info.get('done', False)}
Tick: {self.game_state.last_info.get('tick', 'Unknown')}

Available Actions: {len(self.game_state.env.list_actions()) if self.game_state.env else 0}"""
            
            if self.game_state.move_history:
                history_html = ""
                for move in self.game_state.move_history[-10:]:  # Show last 10 moves
                    history_html += f"[{move['timestamp']}] {move['player']}: {move['move']}<br>"
        
        response = {
            'info': info_text,
            'history': history_html,
            'step': self.game_state.current_step,
            'isRunning': self.game_state.is_running
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def serve_image(self):
        if self.game_state.last_image:
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(self.game_state.last_image)
        else:
            self.send_response(404)
            self.end_headers()
    
    def serve_next_step(self):
        try:
            if not self.game_state.env:
                self.send_json_response({'success': False, 'error': 'Game not started'})
                return
            
            # Check if manual play is stopped
            if self.game_state.manual_play_stopped:
                self.send_json_response({'success': False, 'error': 'Game stopped by user'})
                return
            
            acts = self.game_state.env.list_actions()
            if len(acts) == 0:
                self.send_json_response({'success': False, 'error': 'No actions available'})
                return
            
            # Choose random action
            action_idx = random.randint(0, len(acts) - 1)
            action = acts[action_idx]
            
            # Execute action
            obs, rew, done, info = self.game_state.env.step(action_idx)
            
            # Record move
            player = f"Player {info.get('activeTribeID', 0)}"
            move_desc = action['repr']
            self.game_state.move_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'player': player,
                'move': move_desc,
                'step': self.game_state.current_step
            })
            
            # Update game state
            self.game_state.current_step += 1
            self.game_state.last_info = info
            
            # Get game image
            try:
                game_img = self.game_state.env.render(mode="rgb_image")
                if game_img:
                    # Convert to bytes
                    img_buffer = BytesIO()
                    game_img.save(img_buffer, format='PNG')
                    self.game_state.last_image = img_buffer.getvalue()
                    
                    # Record frame for video
                    if self.game_state.recording:
                        self.record_frame(game_img)
            except Exception as e:
                print(f"Error rendering image: {e}")
            
            # Check if game is completed
            if done:
                self.game_state.game_completed = True
                if self.game_state.recording:
                    self.finish_recording()
            
            self.send_json_response({
                'success': True, 
                'step': self.game_state.current_step,
                'done': done,
                'game_completed': self.game_state.game_completed
            })
            
        except Exception as e:
            print(f"Error in next_step: {e}")
            self.send_json_response({'success': False, 'error': str(e)})
    
    def serve_auto_play_post(self, data):
        steps = data.get('steps', 10)
        speed = data.get('speed', 1.0)
        self.game_state.auto_steps_remaining = steps
        self.game_state.speed = speed
        self.game_state.is_running = True
        
        # Start auto-play in background
        def auto_play_worker():
            while self.game_state.is_running and self.game_state.auto_steps_remaining > 0:
                try:
                    acts = self.game_state.env.list_actions()
                    if len(acts) == 0:
                        break
                    
                    action_idx = random.randint(0, len(acts) - 1)
                    action = acts[action_idx]
                    
                    obs, rew, done, info = self.game_state.env.step(action_idx)
                    
                    player = f"Player {info.get('activeTribeID', 0)}"
                    move_desc = action['repr']
                    self.game_state.move_history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'player': player,
                        'move': move_desc,
                        'step': self.game_state.current_step
                    })
                    
                    self.game_state.current_step += 1
                    self.game_state.last_info = info
                    self.game_state.auto_steps_remaining -= 1
                    
                    # Update image
                    try:
                        game_img = self.game_state.env.render(mode="rgb_image")
                        if game_img:
                            img_buffer = BytesIO()
                            game_img.save(img_buffer, format='PNG')
                            self.game_state.last_image = img_buffer.getvalue()
                    except Exception as e:
                        print(f"Error rendering image in auto-play: {e}")
                    
                    if done:
                        break
                    
                    # Sleep based on speed (speed is in seconds per step)
                    time.sleep(self.game_state.speed)
                    
                except Exception as e:
                    print(f"Error in auto-play: {e}")
                    break
            
            self.game_state.is_running = False
        
        threading.Thread(target=auto_play_worker, daemon=True).start()
        
        self.send_json_response({'success': True})
    
    def serve_stop(self):
        self.game_state.is_running = False
        self.game_state.auto_steps_remaining = 0
        self.game_state.manual_play_stopped = True
        self.save_move_history()
        self.send_json_response({'success': True})
    
    def serve_start_game(self):
        try:
            print("Setting up game environment...")
            
            # Setup environment
            tribes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            os.chdir(tribes_dir)
            
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
            json_jar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "json.jar"))
            cp = os.environ.get("CLASSPATH", "")
            sep = ":"
            os.environ["CLASSPATH"] = sep.join([out_dir, json_jar] + ([cp] if cp else []))
            
            print("Creating environment...")
            self.game_state.env = make_default_env()
            
            print("Loading level...")
            level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levels", "SampleLevel.csv"))
            obs = self.game_state.env.reset(level, seed=42, mode="SCORE")
            
            self.game_state.current_step = 0
            self.game_state.move_history = []
            self.game_state.last_image = None
            self.game_state.last_info = None
            self.game_state.setup_error = None
            self.game_state.game_completed = False
            self.game_state.manual_play_stopped = False
            
            # Initialize video recording
            self.start_recording()
            
            print("Game setup complete!")
            self.send_json_response({'success': True})
            
        except Exception as e:
            print(f"Error setting up game: {e}")
            self.game_state.setup_error = str(e)
            self.send_json_response({'success': False, 'error': str(e)})
    
    def save_move_history(self):
        """Save move history to JSON file"""
        if not self.game_state.move_history:
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_moves_{timestamp}.json"
            
            # Prepare data for JSON
            game_data = {
                'game_info': {
                    'total_steps': self.game_state.current_step,
                    'total_moves': len(self.game_state.move_history),
                    'created_at': datetime.now().isoformat(),
                    'final_scores': self.game_state.last_info.get('scores', []) if self.game_state.last_info else []
                },
                'moves': self.game_state.move_history
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(game_data, f, indent=2)
            
            print(f"Move history saved to {filename}")
            
        except Exception as e:
            print(f"Error saving move history: {e}")
    
    def serve_save_history(self):
        """API endpoint to manually save move history"""
        try:
            self.save_move_history()
            self.send_json_response({'success': True, 'message': 'Move history saved'})
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def start_recording(self):
        """Start video recording"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.game_state.video_filename = f"game_recording_{timestamp}.mp4"
            self.game_state.recording = True
            self.game_state.video_frames = []
            print(f"Started recording: {self.game_state.video_filename}")
        except Exception as e:
            print(f"Error starting recording: {e}")
    
    def record_frame(self, game_img):
        """Record a frame for the video"""
        try:
            # Resize image to standard video size
            frame = game_img.resize((1200, 600), Image.Resampling.LANCZOS)
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            self.game_state.video_frames.append(frame_cv)
        except Exception as e:
            print(f"Error recording frame: {e}")
    
    def finish_recording(self):
        """Finish video recording and save to file"""
        try:
            if not self.game_state.video_frames:
                print("No frames to save")
                return
            
            # Get frame dimensions
            height, width = self.game_state.video_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 2  # 2 frames per second
            video_writer = cv2.VideoWriter(self.game_state.video_filename, fourcc, fps, (width, height))
            
            # Write all frames
            for frame in self.game_state.video_frames:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video saved: {self.game_state.video_filename}")
            
        except Exception as e:
            print(f"Error finishing recording: {e}")
    
    def serve_download_video(self):
        """Serve video file for download"""
        try:
            if not self.game_state.video_filename or not os.path.exists(self.game_state.video_filename):
                self.send_error(404)
                return
            
            with open(self.game_state.video_filename, 'rb') as f:
                video_data = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Disposition', f'attachment; filename="{self.game_state.video_filename}"')
            self.send_header('Content-Length', str(len(video_data)))
            self.end_headers()
            self.wfile.write(video_data)
            
        except Exception as e:
            print(f"Error serving video: {e}")
            self.send_error(500)
    
    def serve_download_json(self):
        """Serve JSON file for download"""
        try:
            # Create a temporary JSON file with current data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_game_moves_{timestamp}.json"
            
            # Prepare data for JSON
            game_data = {
                'game_info': {
                    'total_steps': self.game_state.current_step,
                    'total_moves': len(self.game_state.move_history),
                    'created_at': datetime.now().isoformat(),
                    'final_scores': self.game_state.last_info.get('scores', []) if self.game_state.last_info else []
                },
                'moves': self.game_state.move_history
            }
            
            # Save to temporary file
            with open(temp_filename, 'w') as f:
                json.dump(game_data, f, indent=2)
            
            # Read and serve
            with open(temp_filename, 'rb') as f:
                json_data = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Disposition', f'attachment; filename="game_moves_{timestamp}.json"')
            self.send_header('Content-Length', str(len(json_data)))
            self.end_headers()
            self.wfile.write(json_data)
            
            # Clean up temporary file
            os.remove(temp_filename)
            
        except Exception as e:
            print(f"Error serving JSON: {e}")
            self.send_error(500)
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def create_handler(game_state):
    def handler(*args, **kwargs):
        return SimpleWebHandler(*args, game_state=game_state, **kwargs)
    return handler


def main():
    game_state = GameState()
    
    # Create server
    handler = create_handler(game_state)
    server = HTTPServer(('localhost', 8000), handler)
    
    print("Starting web-based Polytopia Game Visualizer...")
    print("Open your browser and go to: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    # Open browser automatically
    webbrowser.open('http://localhost:8000')
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
