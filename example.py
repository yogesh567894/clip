#!/usr/bin/env python
"""
Example script to demonstrate the AI Video Enhancement Processor.
Processes a sample video from the input_videos directory with different aspect ratios.
"""

import os
import sys
from video_enhancer import VideoEnhancer
import glob

def run_example():
    print("AI Video Enhancement Processor Example")
    print("======================================")
    
    # Check for input videos
    input_dir = "input_videos"
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return 1
    
    # Find all video files in the input directory
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos = []
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not videos:
        print(f"No videos found in '{input_dir}' directory.")
        return 1
    
    # Sort videos by size (smallest first for faster testing)
    videos.sort(key=os.path.getsize)
    
    # List the available videos
    print("\nAvailable videos:")
    for i, video in enumerate(videos):
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"{i+1}. {os.path.basename(video)} ({size_mb:.1f} MB)")
    
    # Ask user to select a video
    try:
        choice = int(input(f"\nSelect a video (1-{len(videos)}) or press Enter for default: ") or "1")
        if choice < 1 or choice > len(videos):
            print(f"Invalid choice. Using first video.")
            choice = 1
    except ValueError:
        print(f"Invalid input. Using first video.")
        choice = 1
    
    selected_video = videos[choice-1]
    print(f"\nSelected: {os.path.basename(selected_video)}")
    
    # Ask for aspect ratio
    aspect_ratios = ["1:1", "4:5", "9:16"]
    print("\nTarget aspect ratios:")
    for i, ratio in enumerate(aspect_ratios):
        print(f"{i+1}. {ratio}")
    
    try:
        ratio_choice = int(input(f"Select an aspect ratio (1-{len(aspect_ratios)}) or press Enter for 9:16: ") or "3")
        if ratio_choice < 1 or ratio_choice > len(aspect_ratios):
            print(f"Invalid choice. Using 9:16.")
            ratio_choice = 3
    except ValueError:
        print(f"Invalid input. Using 9:16.")
        ratio_choice = 3
    
    aspect_ratio = aspect_ratios[ratio_choice-1]
    
    # Ask for quality level
    quality_levels = ["1080p", "720p", "480p"]
    print("\nTarget quality levels:")
    for i, quality in enumerate(quality_levels):
        print(f"{i+1}. {quality}")
    
    try:
        quality_choice = int(input(f"Select a quality level (1-{len(quality_levels)}) or press Enter for 720p: ") or "2")
        if quality_choice < 1 or quality_choice > len(quality_levels):
            print(f"Invalid choice. Using 720p.")
            quality_choice = 2
    except ValueError:
        print(f"Invalid input. Using 720p.")
        quality_choice = 2
    
    quality = quality_levels[quality_choice-1]
    
    # Ask for debug mode
    debug_mode = input("Enable debug visualization? (y/n, default: n): ").lower() == 'y'
    
    print("\nProcessing video with the following settings:")
    print(f"  - Video: {os.path.basename(selected_video)}")
    print(f"  - Target aspect ratio: {aspect_ratio}")
    print(f"  - Target quality: {quality}")
    print(f"  - Debug mode: {'Enabled' if debug_mode else 'Disabled'}")
    
    # Create enhancer and process the video
    enhancer = VideoEnhancer()
    try:
        print("\nStarting video enhancement process...")
        output_path = enhancer.enhance_video(
            selected_video,
            source_language="en-US",
            target_aspect_ratio=aspect_ratio,
            target_quality=quality,
            enable_debug=debug_mode
        )
        print(f"\nProcessing complete!")
        print(f"Enhanced video saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\nError processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    sys.exit(run_example()) 