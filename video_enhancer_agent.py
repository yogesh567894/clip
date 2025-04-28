#!/usr/bin/env python
"""
AI Video Enhancement Processor Agent
A unified driver for video enhancement with intelligent reframing and caption burning
"""

import os
import sys
import argparse
import glob
import time
import json
import subprocess
from moviepy.editor import VideoFileClip
from video_enhancer import VideoEnhancer, ASPECT_RATIOS, QUALITY_LEVELS, DEFAULT_CAPTION_STYLE

class VideoEnhancerAgent:
    """Agent that orchestrates the video enhancement process"""
    
    def __init__(self):
        self.enhancer = VideoEnhancer()
        self.input_videos = []
        self.output_dir = "output_videos"
        self.clips_dir = "output_clips"
        
        # Ensure output directories exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.clips_dir):
            os.makedirs(self.clips_dir)
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'enhancer'):
            self.enhancer.cleanup()
    
    def find_videos(self, path_or_pattern):
        """Find videos based on path or glob pattern"""
        if os.path.isfile(path_or_pattern) and self._is_video_file(path_or_pattern):
            self.input_videos = [os.path.abspath(path_or_pattern)]
        elif os.path.isdir(path_or_pattern):
            # Search for videos in directory
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
            videos = []
            for ext in video_extensions:
                videos.extend(glob.glob(os.path.join(path_or_pattern, f"*{ext}")))
            self.input_videos = [os.path.abspath(v) for v in videos]
        else:
            # Treat as glob pattern
            self.input_videos = [os.path.abspath(v) for v in glob.glob(path_or_pattern)]
        
        # Sort by size
        self.input_videos.sort(key=os.path.getsize)
        return len(self.input_videos) > 0
    
    def _is_video_file(self, path):
        """Check if file has video extension"""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        _, ext = os.path.splitext(path)
        return ext.lower() in video_extensions
    
    def list_videos(self):
        """List available videos with info"""
        print(f"\nFound {len(self.input_videos)} video(s):")
        for i, video in enumerate(self.input_videos):
            size_mb = os.path.getsize(video) / (1024 * 1024)
            print(f"{i+1}. {os.path.basename(video)} ({size_mb:.1f} MB)")
    
    def process_single_video(self, video_path, source_language, aspect_ratio, quality, 
                           transcript=None, caption_style=None, enable_debug=False):
        """Process a single video with the enhancer"""
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  - Target aspect ratio: {aspect_ratio}")
        print(f"  - Target quality: {quality}")
        
        start_time = time.time()
        
        try:
            output_path = self.enhancer.enhance_video(
                video_path,
                source_language,
                aspect_ratio,
                quality,
                provided_transcript=transcript,
                caption_style=caption_style,
                enable_debug=enable_debug
            )
            
            elapsed = time.time() - start_time
            print(f"\n✓ Completed in {elapsed:.1f} seconds")
            print(f"  Enhanced video saved to: {output_path}")
            
            return {
                "input": video_path,
                "output": output_path,
                "success": True,
                "time": elapsed
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "input": video_path,
                "error": str(e),
                "success": False,
                "time": elapsed
            }
    
    def extract_clip(self, video_path, start_time, end_time, output_path=None):
        """Extract a clip from a video file using FFmpeg"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output filename if not provided
        if output_path is None:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{basename}_clip_{int(start_time)}s_to_{int(end_time)}s.mp4"
            output_path = os.path.join(self.clips_dir, output_filename)
        
        # Check for ffmpeg in the enhancer
        ffmpeg_path = self.enhancer._get_ffmpeg_path()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or set FFMPEG_PATH environment variable.")
        
        # Use FFmpeg to extract the clip
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output file if it exists
            "-i", video_path,
            "-ss", str(start_time),  # Start time
            "-to", str(end_time),    # End time
            "-c:v", "libx264",       # Video codec
            "-c:a", "aac",           # Audio codec
            "-strict", "experimental",
            output_path
        ]
        
        print(f"Extracting clip from {start_time}s to {end_time}s...")
        subprocess.run(cmd, check=True)
        
        # Verify the clip was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to create clip: {output_path}")
        
        print(f"Clip extracted to: {output_path}")
        return output_path
    
    def extract_clips_interactive(self, video_path):
        """Interactive mode for extracting clips from a video"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
        
        # Get video duration
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                print(f"\nVideo: {os.path.basename(video_path)}")
                print(f"Duration: {int(duration // 60)}m {int(duration % 60)}s ({duration:.2f}s)")
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return False
        
        # Ask how many clips to extract
        try:
            num_clips = int(input("\nHow many clips would you like to extract? "))
            if num_clips <= 0:
                print("Invalid number of clips.")
                return False
        except ValueError:
            print("Invalid input. Please enter a number.")
            return False
        
        clips = []
        
        # Get clip timestamps
        for i in range(num_clips):
            print(f"\nClip {i+1}/{num_clips}:")
            
            # Get start time
            while True:
                try:
                    start_input = input(f"Start time (seconds, max {duration:.2f}): ")
                    start_time = float(start_input)
                    if 0 <= start_time < duration:
                        break
                    print(f"Start time must be between 0 and {duration:.2f} seconds.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Get end time
            while True:
                try:
                    end_input = input(f"End time (seconds, > {start_time:.2f}, max {duration:.2f}): ")
                    end_time = float(end_input)
                    if start_time < end_time <= duration:
                        break
                    print(f"End time must be between {start_time:.2f} and {duration:.2f} seconds.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Add clip info
            clips.append({
                "start": start_time,
                "end": end_time
            })
        
        # Extract clips
        extracted_paths = []
        for i, clip_info in enumerate(clips):
            try:
                start_time = clip_info["start"]
                end_time = clip_info["end"]
                
                # Extract the clip
                output_path = self.extract_clip(video_path, start_time, end_time)
                extracted_paths.append(output_path)
                
                print(f"✓ Extracted clip {i+1}/{num_clips}")
            except Exception as e:
                print(f"✗ Failed to extract clip {i+1}/{num_clips}: {e}")
        
        if extracted_paths:
            print(f"\nSuccessfully extracted {len(extracted_paths)}/{num_clips} clips:")
            for path in extracted_paths:
                print(f"  - {os.path.basename(path)}")
            return True
        else:
            print("\nFailed to extract any clips.")
            return False
    
    def batch_process(self, source_language, aspect_ratio, quality, caption_style=None):
        """Process all input videos with the same settings"""
        if not self.input_videos:
            print("No videos to process.")
            return False
        
        results = []
        total_start_time = time.time()
        
        for i, video in enumerate(self.input_videos):
            print(f"\n[{i+1}/{len(self.input_videos)}] Processing: {os.path.basename(video)}")
            
            result = self.process_single_video(
                video,
                source_language,
                aspect_ratio,
                quality,
                caption_style=caption_style
            )
            
            results.append(result)
        
        # Print summary
        total_elapsed = time.time() - total_start_time
        success_count = sum(1 for r in results if r["success"])
        
        print("\n" + "="*60)
        print(f"Batch Processing Summary")
        print(f"Completed: {success_count}/{len(self.input_videos)} videos")
        print(f"Total time: {total_elapsed:.1f} seconds")
        print("="*60)
        
        if success_count > 0:
            print("\nSuccessfully processed videos:")
            for i, result in enumerate([r for r in results if r["success"]]):
                print(f"{i+1}. {os.path.basename(result['input'])} → {os.path.basename(result['output'])} ({result['time']:.1f}s)")
                
            # Ask if user wants to extract clips from any of the processed videos
            extract_clips = input("\nWould you like to extract clips from any of the processed videos? (y/n): ").lower() == 'y'
            if extract_clips:
                # List successful outputs
                successful_outputs = [r["output"] for r in results if r["success"]]
                print("\nProcessed videos:")
                for i, output in enumerate(successful_outputs):
                    print(f"{i+1}. {os.path.basename(output)}")
                
                # Get video selection
                try:
                    choice = int(input(f"\nSelect a video (1-{len(successful_outputs)}): "))
                    if 1 <= choice <= len(successful_outputs):
                        selected_video = successful_outputs[choice-1]
                        self.extract_clips_interactive(selected_video)
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        if success_count < len(self.input_videos):
            print("\nFailed videos:")
            for i, result in enumerate([r for r in results if not r["success"]]):
                print(f"{i+1}. {os.path.basename(result['input'])}: {result['error']}")
        
        return success_count == len(self.input_videos)
    
    def interactive_mode(self):
        """Run interactive selection for video enhancement"""
        print("AI Video Enhancement Processor")
        print("=============================")
        
        # Check if user wants to extract clips from an existing video
        extract_from_existing = input("Extract clips from an existing processed video? (y/n, default: n): ").lower() == 'y'
        
        if extract_from_existing:
            # Find videos in output directory
            if not os.path.exists(self.output_dir):
                print(f"Output directory '{self.output_dir}' not found.")
                return False
            
            output_videos = []
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                output_videos.extend(glob.glob(os.path.join(self.output_dir, f"*{ext}")))
            
            if not output_videos:
                print(f"No processed videos found in '{self.output_dir}' directory.")
                return False
            
            # List available processed videos
            print(f"\nFound {len(output_videos)} processed video(s):")
            for i, video in enumerate(output_videos):
                size_mb = os.path.getsize(video) / (1024 * 1024)
                print(f"{i+1}. {os.path.basename(video)} ({size_mb:.1f} MB)")
            
            # Select video
            try:
                choice = int(input(f"\nSelect a video (1-{len(output_videos)}): "))
                if 1 <= choice <= len(output_videos):
                    selected_video = output_videos[choice-1]
                    return self.extract_clips_interactive(selected_video)
                else:
                    print("Invalid choice.")
                    return False
            except ValueError:
                print("Invalid input.")
                return False
        
        # Regular video enhancement process
        # Get videos from input_videos directory as default
        input_dir = "input_videos"
        if os.path.exists(input_dir):
            if self.find_videos(input_dir):
                self.list_videos()
            else:
                print(f"No videos found in {input_dir} directory.")
                return False
        else:
            print(f"Input directory '{input_dir}' not found.")
            video_path = input("Enter path to a video file or directory: ")
            if not self.find_videos(video_path):
                print("No videos found.")
                return False
            self.list_videos()
        
        # Select video if multiple available
        selected_idx = 0
        if len(self.input_videos) > 1:
            try:
                choice = int(input(f"\nSelect a video (1-{len(self.input_videos)}) or press Enter for default: ") or "1")
                if 1 <= choice <= len(self.input_videos):
                    selected_idx = choice - 1
                else:
                    print("Invalid choice. Using first video.")
            except ValueError:
                print("Invalid input. Using first video.")
        
        selected_video = self.input_videos[selected_idx]
        print(f"\nSelected: {os.path.basename(selected_video)}")
        
        # Select aspect ratio
        aspect_ratios = list(ASPECT_RATIOS.keys())
        print("\nTarget aspect ratios:")
        for i, ratio in enumerate(aspect_ratios):
            print(f"{i+1}. {ratio}")
        
        aspect_idx = 2  # Default to 9:16
        try:
            ratio_choice = int(input(f"Select an aspect ratio (1-{len(aspect_ratios)}) or press Enter for 9:16: ") or "3")
            if 1 <= ratio_choice <= len(aspect_ratios):
                aspect_idx = ratio_choice - 1
            else:
                print("Invalid choice. Using 9:16.")
        except ValueError:
            print("Invalid input. Using 9:16.")
        
        aspect_ratio = aspect_ratios[aspect_idx]
        
        # Select quality level
        quality_levels = list(QUALITY_LEVELS.keys())
        print("\nTarget quality levels:")
        for i, quality in enumerate(quality_levels):
            print(f"{i+1}. {quality}")
        
        quality_idx = 1  # Default to 720p
        try:
            quality_choice = int(input(f"Select a quality level (1-{len(quality_levels)}) or press Enter for 720p: ") or "2")
            if 1 <= quality_choice <= len(quality_levels):
                quality_idx = quality_choice - 1
            else:
                print("Invalid choice. Using 720p.")
        except ValueError:
            print("Invalid input. Using 720p.")
        
        quality = quality_levels[quality_idx]
        
        # Caption settings
        customize_captions = input("\nCustomize caption styling? (y/n, default: n): ").lower() == 'y'
        caption_style = DEFAULT_CAPTION_STYLE.copy()
        
        if customize_captions:
            caption_style["font"] = input(f"Font (default: {caption_style['font']}): ") or caption_style["font"]
            
            try:
                size = int(input(f"Font size (default: {caption_style['fontsize']}): ") or caption_style["fontsize"])
                caption_style["fontsize"] = size
            except ValueError:
                print(f"Invalid font size. Using default: {caption_style['fontsize']}")
            
            caption_style["color"] = input(f"Text color (default: {caption_style['color']}): ") or caption_style["color"]
            caption_style["bg_color"] = input(f"Background color (default: {caption_style['bg_color']}): ") or caption_style["bg_color"]
            
            try:
                opacity = float(input(f"Background opacity (0-1, default: {caption_style['bg_opacity']}): ") or caption_style["bg_opacity"])
                caption_style["bg_opacity"] = max(0, min(1, opacity))  # Clamp between 0 and 1
            except ValueError:
                print(f"Invalid opacity. Using default: {caption_style['bg_opacity']}")
            
            positions = ["top", "middle", "bottom"]
            print("Caption positions:")
            for i, pos in enumerate(positions):
                print(f"{i+1}. {pos}")
            
            try:
                pos_choice = int(input(f"Select position (1-3, default: 3 for bottom): ") or "3")
                if 1 <= pos_choice <= len(positions):
                    caption_style["position"] = positions[pos_choice-1]
                else:
                    print(f"Invalid choice. Using default: {caption_style['position']}")
            except ValueError:
                print(f"Invalid input. Using default: {caption_style['position']}")
        
        # Debug mode
        debug_mode = input("\nEnable debug visualization? (y/n, default: n): ").lower() == 'y'
        
        # Source language
        source_language = input("\nSource language code (default: en-US): ") or "en-US"
        
        # Transcript option
        use_transcript = input("\nUse pre-existing transcript? (y/n, default: n): ").lower() == 'y'
        transcript = None
        
        if use_transcript:
            transcript_path = input("Path to transcript file (JSON format): ")
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = json.load(f)
                    print(f"Loaded transcript with {len(transcript)} segments")
                except Exception as e:
                    print(f"Error loading transcript: {e}")
                    print("Will use automatic transcription instead")
            else:
                print(f"Transcript file not found: {transcript_path}")
                print("Will use automatic transcription instead")
        
        # Process the video
        print("\nProcessing video with the following settings:")
        print(f"  - Video: {os.path.basename(selected_video)}")
        print(f"  - Target aspect ratio: {aspect_ratio}")
        print(f"  - Target quality: {quality}")
        print(f"  - Source language: {source_language}")
        print(f"  - Debug mode: {'Enabled' if debug_mode else 'Disabled'}")
        
        result = self.process_single_video(
            selected_video,
            source_language,
            aspect_ratio,
            quality,
            transcript=transcript,
            caption_style=caption_style,
            enable_debug=debug_mode
        )
        
        # Ask if user wants to extract clips from the processed video
        if result["success"]:
            extract_clips = input("\nWould you like to extract clips from the processed video? (y/n): ").lower() == 'y'
            if extract_clips:
                self.extract_clips_interactive(result["output"])
        
        return result["success"]

def main():
    """Main entry point for the video enhancer agent"""
    parser = argparse.ArgumentParser(
        description="AI Video Enhancement Processor - Convert aspect ratios with intelligent reframing and add burned-in captions"
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-i", "--interactive", action="store_true", 
                           help="Run in interactive mode with prompts for all options")
    mode_group.add_argument("-b", "--batch", action="store_true",
                           help="Process multiple videos with the same settings")
    mode_group.add_argument("-c", "--clip", action="store_true",
                           help="Extract clips from an existing video")
    
    # Video input options
    parser.add_argument("input", nargs="?", default="input_videos",
                       help="Video file, directory, or glob pattern (default: 'input_videos' directory)")
    
    # Video processing options
    parser.add_argument("--aspect-ratio", choices=list(ASPECT_RATIOS.keys()), default="9:16",
                       help="Target aspect ratio for the output video (default: 9:16)")
    parser.add_argument("--quality", choices=list(QUALITY_LEVELS.keys()), default="720p",
                       help="Target quality level for the output (default: 720p)")
    parser.add_argument("--source-language", default="en-US",
                       help="Source language code for captioning (default: en-US)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug visualization of subject tracking")
    
    # Caption customization options
    caption_group = parser.add_argument_group("Caption Styling")
    caption_group.add_argument("--caption-font", default=DEFAULT_CAPTION_STYLE["font"],
                              help=f"Font to use for captions (default: {DEFAULT_CAPTION_STYLE['font']})")
    caption_group.add_argument("--caption-size", type=int, default=DEFAULT_CAPTION_STYLE["fontsize"],
                              help=f"Font size for captions (default: {DEFAULT_CAPTION_STYLE['fontsize']})")
    caption_group.add_argument("--caption-color", default=DEFAULT_CAPTION_STYLE["color"],
                              help=f"Text color for captions (default: {DEFAULT_CAPTION_STYLE['color']})")
    caption_group.add_argument("--caption-bg-color", default=DEFAULT_CAPTION_STYLE["bg_color"],
                              help=f"Background color for captions (default: {DEFAULT_CAPTION_STYLE['bg_color']})")
    caption_group.add_argument("--caption-bg-opacity", type=float, default=DEFAULT_CAPTION_STYLE["bg_opacity"],
                              help=f"Background opacity for captions (0-1, default: {DEFAULT_CAPTION_STYLE['bg_opacity']})")
    caption_group.add_argument("--caption-position", choices=["top", "middle", "bottom"], 
                              default=DEFAULT_CAPTION_STYLE["position"],
                              help=f"Vertical position of captions (default: {DEFAULT_CAPTION_STYLE['position']})")
    
    # Transcript option
    parser.add_argument("--transcript", 
                       help="Path to pre-existing transcript in JSON format (skip transcription)")
    
    # Clip extraction options
    clip_group = parser.add_argument_group("Clip Extraction")
    clip_group.add_argument("--start", type=float,
                           help="Start time for clip extraction (in seconds)")
    clip_group.add_argument("--end", type=float,
                           help="End time for clip extraction (in seconds)")
    clip_group.add_argument("--output-clip",
                           help="Output filename for the extracted clip (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create the agent
    agent = VideoEnhancerAgent()
    
    try:
        # Interactive mode
        if args.interactive:
            success = agent.interactive_mode()
            return 0 if success else 1
        
        # Clip extraction mode
        if args.clip:
            if not os.path.isfile(args.input) or not agent._is_video_file(args.input):
                print(f"Error: {args.input} is not a valid video file.")
                return 1
            
            if args.start is not None and args.end is not None:
                # Direct clip extraction with start and end times
                try:
                    agent.extract_clip(args.input, args.start, args.end, args.output_clip)
                    return 0
                except Exception as e:
                    print(f"Error extracting clip: {e}")
                    return 1
            else:
                # Interactive clip extraction
                success = agent.extract_clips_interactive(args.input)
                return 0 if success else 1
        
        # Find videos from input pattern
        if not agent.find_videos(args.input):
            print(f"No videos found matching: {args.input}")
            return 1
        
        agent.list_videos()
        
        # Load transcript if provided
        transcript = None
        if args.transcript:
            try:
                with open(args.transcript, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
                print(f"Loaded transcript from {args.transcript}")
            except Exception as e:
                print(f"Error loading transcript: {e}")
                print("Proceeding with automatic transcription...")
        
        # Set up caption styling
        caption_style = {
            "font": args.caption_font,
            "fontsize": args.caption_size,
            "color": args.caption_color,
            "bg_color": args.caption_bg_color,
            "bg_opacity": args.caption_bg_opacity,
            "position": args.caption_position
        }
        
        # Batch mode
        if args.batch:
            success = agent.batch_process(
                args.source_language,
                args.aspect_ratio,
                args.quality,
                caption_style=caption_style
            )
            return 0 if success else 1
        
        # Single video mode (default)
        if len(agent.input_videos) == 1:
            result = agent.process_single_video(
                agent.input_videos[0],
                args.source_language,
                args.aspect_ratio,
                args.quality,
                transcript=transcript,
                caption_style=caption_style,
                enable_debug=args.debug
            )
            return 0 if result["success"] else 1
        else:
            print("Multiple videos found. Use -b/--batch to process all or -i/--interactive to select one.")
            return 1
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 