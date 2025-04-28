#!/usr/bin/env python
"""
Batch processing script for the AI Video Enhancement Processor.
Processes multiple videos with the same settings.
"""

import os
import sys
import glob
import argparse
import time
from video_enhancer import VideoEnhancer

def batch_process(input_pattern, aspect_ratio, quality, source_language="en-US"):
    """Process multiple videos matching the input pattern"""
    # Find all videos matching the pattern
    videos = glob.glob(input_pattern)
    
    if not videos:
        print(f"No videos found matching pattern: {input_pattern}")
        return 1
    
    print(f"Found {len(videos)} videos to process:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {os.path.basename(video)}")
    
    # Create output directory
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each video
    enhancer = VideoEnhancer()
    results = []
    total_start_time = time.time()
    
    try:
        for i, video in enumerate(videos):
            print(f"\nProcessing video {i+1}/{len(videos)}: {os.path.basename(video)}")
            start_time = time.time()
            
            try:
                output_path = enhancer.enhance_video(
                    video,
                    source_language=source_language,
                    target_aspect_ratio=aspect_ratio,
                    target_quality=quality
                )
                
                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed:.1f} seconds: {os.path.basename(output_path)}")
                
                results.append({
                    "input": video,
                    "output": output_path,
                    "success": True,
                    "time": elapsed
                })
            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({
                    "input": video,
                    "error": str(e),
                    "success": False
                })
                
    finally:
        enhancer.cleanup()
    
    # Print summary
    total_elapsed = time.time() - total_start_time
    success_count = sum(1 for r in results if r['success'])
    
    print("\n" + "="*60)
    print(f"Batch Processing Summary")
    print(f"Completed: {success_count}/{len(videos)} videos")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print("="*60)
    
    # Print successful conversions
    if success_count > 0:
        print("\nSuccessfully processed videos:")
        for i, result in enumerate([r for r in results if r['success']]):
            print(f"{i+1}. {os.path.basename(result['input'])} → {os.path.basename(result['output'])} ({result['time']:.1f}s)")
    
    # Print failures
    if success_count < len(videos):
        print("\nFailed videos:")
        for i, result in enumerate([r for r in results if not r['success']]):
            print(f"{i+1}. {os.path.basename(result['input'])}: {result['error']}")
    
    return 0 if success_count == len(videos) else 1

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with the AI Video Enhancement Processor")
    parser.add_argument("input_pattern", help="Glob pattern for input videos (e.g., 'input_videos/*.mp4')")
    parser.add_argument("--aspect-ratio", choices=["1:1", "4:5", "9:16"], default="9:16", 
                        help="Target aspect ratio for all videos")
    parser.add_argument("--quality", choices=["1080p", "720p", "480p"], default="720p", 
                        help="Target quality level for all videos")
    parser.add_argument("--source-language", default="en-US", 
                        help="Source language code for captioning (e.g., en-US, es-ES)")
    
    args = parser.parse_args()
    
    return batch_process(
        args.input_pattern, 
        args.aspect_ratio, 
        args.quality, 
        args.source_language
    )

if __name__ == "__main__":
    sys.exit(main()) 