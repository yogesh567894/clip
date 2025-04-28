#!/usr/bin/env python
"""
MP4 Fixer - Convert incompatible MP4 files to a more compatible format
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path

def find_ffmpeg():
    """Find ffmpeg executable on the system"""
    # Check if FFMPEG_PATH environment variable is set
    ffmpeg_env_path = os.environ.get('FFMPEG_PATH')
    if ffmpeg_env_path and os.path.exists(ffmpeg_env_path):
        return ffmpeg_env_path
        
    # Check common locations for FFmpeg
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # Current directory with version-specific path
        os.path.join(current_dir, "ffmpeg-2025-04-21-git-9e1162bdf1-essentials_build", "bin", "ffmpeg.exe"),
        # Just in case there's a generic ffmpeg folder
        os.path.join(current_dir, "ffmpeg", "bin", "ffmpeg.exe"),
        # For legacy path structure
        os.path.join(current_dir, "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        # Common Windows installation paths
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe"
    ]
    
    # Find the first valid path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Check if 'ffmpeg' command is in PATH
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(["where", "ffmpeg"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
        else:  # Linux/macOS
            result = subprocess.run(["which", "ffmpeg"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
        
    print("Error: FFmpeg not found. Please set FFMPEG_PATH environment variable.")
    return None
    
def fix_video(input_file, output_file=None, crf=18):
    """Fix an MP4 file to make it more compatible with Windows Media Player and other players"""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
        
    # If no output file specified, create one with "_fixed" appended
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    # Find ffmpeg
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return False
    
    print(f"Converting: {input_file}")
    print(f"Output to: {output_file}")
    
    try:
        # Run FFmpeg with very compatible settings
        cmd = [
            ffmpeg_path,
            "-i", input_file,
            "-c:v", "libx264",             # Use H.264 codec
            "-profile:v", "baseline",      # Most compatible H.264 profile
            "-level", "3.0",               # Compatible level
            "-preset", "medium",           # Balance between speed and quality
            "-crf", str(crf),              # Quality level (lower = better quality)
            "-maxrate", "5M",              # Maximum bitrate
            "-bufsize", "5M",              # Buffer size
            "-pix_fmt", "yuv420p",         # Pixel format (required for compatibility)
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
            "-c:a", "aac",                 # Audio codec
            "-b:a", "192k",                # Audio bitrate
            "-ar", "44100",                # Audio sample rate
            "-movflags", "+faststart",     # Put MOOV atom at the beginning
            "-y",                          # Overwrite output file
            output_file
        ]
        
        # Run the command
        print("Running FFmpeg conversion...")
        subprocess.run(cmd, check=True)
        
        print(f"\nConversion complete!")
        print(f"Fixed video saved to: {output_file}")
        
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main function for the MP4 fixer utility"""
    parser = argparse.ArgumentParser(
        description="MP4 Fixer - Convert incompatible MP4 files to a more compatible format"
    )
    
    # Input options
    parser.add_argument("input", nargs="?", help="Input MP4 file or directory to fix")
    parser.add_argument("-o", "--output", help="Output file path (for single file only)")
    parser.add_argument("-q", "--quality", type=int, default=18, 
                        help="Quality level (CRF value, 0-51, lower is better, default: 18)")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Process directories recursively")
    parser.add_argument("-p", "--pattern", default="*.mp4", 
                        help="File pattern to match when processing directories (default: *.mp4)")
    
    args = parser.parse_args()
    
    # No input specified, show usage and find recent videos
    if not args.input:
        # Try to find videos in output_videos directory
        output_dir = "output_videos"
        if os.path.exists(output_dir):
            mp4_files = list(glob.glob(os.path.join(output_dir, "*.mp4")))
            if mp4_files:
                # Sort by modification time (newest first)
                mp4_files.sort(key=os.path.getmtime, reverse=True)
                print("\nRecent MP4 files found in output_videos directory:")
                for i, file in enumerate(mp4_files[:5]):  # Show up to 5 most recent files
                    file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                    print(f"{i+1}. {os.path.basename(file)} ({file_size:.1f} MB)")
                
                try:
                    choice = input("\nSelect a file to fix (or press Enter to exit): ")
                    if choice.isdigit() and 1 <= int(choice) <= len(mp4_files[:5]):
                        selected_file = mp4_files[int(choice)-1]
                        fix_video(selected_file, crf=args.quality)
                        return 0
                    else:
                        print("No valid selection. Exiting.")
                        return 0
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                    return 1
        
        parser.print_help()
        return 1
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        # Process directory
        pattern = os.path.join(args.input, "**", args.pattern) if args.recursive else os.path.join(args.input, args.pattern)
        files = glob.glob(pattern, recursive=args.recursive)
        
        if not files:
            print(f"No files matching pattern '{args.pattern}' found in '{args.input}'.")
            return 1
            
        print(f"Found {len(files)} file(s) to process.")
        
        success_count = 0
        for file in files:
            if fix_video(file, crf=args.quality):
                success_count += 1
                
        print(f"\nProcessed {success_count} of {len(files)} files successfully.")
        return 0 if success_count > 0 else 1
        
    elif os.path.isfile(args.input):
        # Process single file
        result = fix_video(args.input, args.output, crf=args.quality)
        return 0 if result else 1
        
    else:
        print(f"Error: Input '{args.input}' is not a valid file or directory.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 