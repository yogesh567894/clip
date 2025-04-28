#!/usr/bin/env python
"""
MP4 Fixer Compact - Convert incompatible MP4 files to smaller, compatible files
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
        os.path.join(current_dir, "ffmpeg-2025-04-21-git-9e1162bdf1-essentials_build", "bin", "ffmpeg.exe"),
        os.path.join(current_dir, "ffmpeg", "bin", "ffmpeg.exe"),
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
        
    print("Error: FFmpeg not found.")
    return None
    
def fix_video(input_file, output_file=None, crf=28):
    """Fix an MP4 file to make it more compatible and smaller"""
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
    
    # Get output directory and create it if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    # Check disk space before starting (Windows only)
    if os.name == 'nt':
        try:
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(output_dir), None, None, ctypes.pointer(free_bytes))
            free_mb = free_bytes.value / (1024 * 1024)
            
            # Rough estimate of needed space (input file size * 0.5)
            input_size_mb = os.path.getsize(input_file) / (1024 * 1024)
            needed_mb = input_size_mb * 0.5
            
            if free_mb < needed_mb:
                print(f"Warning: Low disk space. Free: {free_mb:.1f}MB, Needed: {needed_mb:.1f}MB")
                print("Using more aggressive compression to save space")
                # Use higher CRF for smaller files
                crf = min(crf + 4, 35)  # Increase CRF but cap at 35
        except:
            print("Could not check disk space, proceeding anyway")
    
    print(f"Converting: {input_file}")
    print(f"Output to: {output_file}")
    print(f"Using compression level: {crf} (higher = smaller files)")
    
    try:
        # Calculate a smaller target resolution (half the original)
        try:
            probe_cmd = [
                ffmpeg_path,
                "-i", input_file,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0"
            ]
            
            probe_result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if probe_result.returncode == 0 and probe_result.stdout:
                width, height = map(int, probe_result.stdout.strip().split(','))
                # Target half resolution but ensure even numbers
                target_width = (width // 2) + (width // 2) % 2
                target_height = (height // 2) + (height // 2) % 2
                scale_filter = f"scale={target_width}:{target_height}"
            else:
                # Default to generic even dimension filter if probing fails
                scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        except:
            # Default filter just ensures even dimensions
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        
        # Run FFmpeg with very compatible but highly compressed settings
        cmd = [
            ffmpeg_path,
            "-i", input_file,
            "-c:v", "libx264",             # Use H.264 codec
            "-profile:v", "baseline",      # Most compatible H.264 profile
            "-level", "3.0",               # Compatible level
            "-preset", "fast",             # Faster encoding
            "-crf", str(crf),              # Quality level (higher = smaller files)
            "-maxrate", "1M",              # Lower maximum bitrate
            "-bufsize", "2M",              # Buffer size
            "-pix_fmt", "yuv420p",         # Pixel format (required for compatibility)
            "-vf", scale_filter,           # Scale down to save space
            "-c:a", "aac",                 # Audio codec
            "-b:a", "96k",                 # Lower audio bitrate
            "-ac", "2",                    # 2 audio channels (stereo)
            "-ar", "44100",                # Audio sample rate
            "-movflags", "+faststart",     # Put MOOV atom at the beginning
            "-y",                          # Overwrite output file
            output_file
        ]
        
        # Run the command
        print("Running FFmpeg conversion...")
        subprocess.run(cmd, check=True)
        
        # Check if the file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            orig_size = os.path.getsize(input_file) / (1024 * 1024)
            new_size = os.path.getsize(output_file) / (1024 * 1024)
            reduction = (1 - (new_size / orig_size)) * 100
            
            print(f"\nConversion complete!")
            print(f"Original size: {orig_size:.1f}MB")
            print(f"New size: {new_size:.1f}MB ({reduction:.1f}% smaller)")
            print(f"Fixed video saved to: {output_file}")
            
            # Try to open the file with the default player
            try:
                if os.name == 'nt':
                    os.startfile(output_file)
                else:
                    # For Unix-like systems
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', output_file])
                    else:  # Linux
                        subprocess.run(['xdg-open', output_file])
            except:
                print("Could not automatically open the file")
            
            return output_file
        else:
            print("Error: Output file was not created properly.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="MP4 Fixer Compact - Make videos smaller and compatible"
    )
    
    parser.add_argument("input", nargs="?", help="Input MP4 file or directory")
    parser.add_argument("-o", "--output", help="Output file path (for single file)")
    parser.add_argument("-q", "--quality", type=int, default=28, 
                       help="Quality level (CRF, 0-51, higher=smaller, default: 28)")
    
    args = parser.parse_args()
    
    # No input specified, look for recent videos
    if not args.input:
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
                        # Create output in a different directory
                        fixed_dir = "fixed_videos"
                        if not os.path.exists(fixed_dir):
                            os.makedirs(fixed_dir)
                        
                        output_file = os.path.join(fixed_dir, f"{os.path.basename(selected_file)}")
                        fix_video(selected_file, output_file, crf=args.quality)
                        return 0
                    else:
                        print("No valid selection. Exiting.")
                        return 0
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                    return 1
        
        parser.print_help()
        return 1
    
    # Process a single file
    if os.path.isfile(args.input):
        result = fix_video(args.input, args.output, crf=args.quality)
        return 0 if result else 1
    else:
        print(f"Error: Input '{args.input}' is not a valid file.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 