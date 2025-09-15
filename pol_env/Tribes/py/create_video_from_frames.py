#!/usr/bin/env python3
"""
Script to create videos from frame directories.
This demonstrates the new folder structure where frames are saved separately from videos.
"""

import os
import glob
import imageio.v2 as imageio
import argparse
from datetime import datetime

def create_video_from_frames(frames_dir, output_path=None, fps=2):
    """
    Create a video from PNG frames in a directory.
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_path: Output video path (optional, will auto-generate if not provided)
        fps: Frames per second for the video
    """
    # Get all frame files and sort them
    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"videos/game_recording_{timestamp}.mp4"
    
    # Ensure videos directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Creating video from {len(frame_files)} frames...")
    print(f"Input frames: {frames_dir}")
    print(f"Output video: {output_path}")
    print(f"FPS: {fps}")
    
    try:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frame_files:
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
        
        print(f"✅ Video created successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create video from game frames")
    parser.add_argument("frames_dir", help="Directory containing frame_*.png files")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-f", "--fps", type=int, default=2, help="Frames per second (default: 2)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frames_dir):
        print(f"❌ Frames directory does not exist: {args.frames_dir}")
        return 1
    
    success = create_video_from_frames(args.frames_dir, args.output, args.fps)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
