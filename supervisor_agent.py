import os
import sys
import subprocess
import argparse
import json
import shutil
import datetime
from typing import List, Dict, Any, Optional

# Configuration for the workflow
CONFIG = {
    "input_dir": "output_videos",     # Where agent_app.py saves extracted clips (and where caption_enhancer.py reads from)
    "clips_dir": "output_videos",     # Where agent_app.py saves extracted clips
    "output_dir": "output_clips",     # Where caption_enhancer.py saves final enhanced clips
    "log_file": "workflow_log.txt"    # Log file name changed to reflect workflow purpose
}

class WorkflowDriver:
    """
    WorkflowDriver orchestrates the video processing workflow:
    1. Upload videos to input_videos folder
    2. Use agent_app.py to extract clips based on prompts (save to output_videos)
    3. Use caption_enhancer.py to add captions and change aspect ratio (save to output_clips)
    """
    
    def __init__(self):
        self.ensure_directories()
        self.log("Workflow driver initialized")
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        for dir_path in [CONFIG["input_dir"], CONFIG["clips_dir"], CONFIG["output_dir"]]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                self.log(f"Created directory: {dir_path}")
    
    def extract_clips(self, video_path: str = None, query: str = "Create engaging clips", 
                     num_clips: int = 3) -> Dict[str, Any]:
        """
        Extract clips from a video using agent_app.py and save to output_videos
        """
        # If no video path provided, use all videos in input directory
        if not video_path:
            videos = self.search_videos("input_videos")  # Search in input_videos directory
            if videos.get("error"):
                return videos
            if videos.get("count", 0) == 0:
                return {"error": "No videos found in input_videos directory"}
            
            results = []
            for video in videos["videos"]:
                result = self.extract_clips(video["path"], query, num_clips)
                results.append(result)
            
            return {
                "success": True,
                "message": f"Processed {len(results)} videos",
                "results": results
            }
        
        self.log(f"Extracting {num_clips} clips from {video_path} with query: '{query}'")
        
        # Check that the video exists
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        # Copy the video to input_videos if it's not already there
        video_filename = os.path.basename(video_path)
        input_video_path = os.path.join("input_videos", video_filename)  # Use input_videos directory
        
        if os.path.abspath(video_path) != os.path.abspath(input_video_path):
            # Make sure input_videos directory exists
            if not os.path.exists("input_videos"):
                os.makedirs("input_videos")
            shutil.copy2(video_path, input_video_path)
            self.log(f"Copied video to {input_video_path}")
            video_path = input_video_path
        
        # Prepare command
        cmd = [
            sys.executable,
            "agent_app.py",
            "--video", video_filename,
            "--query", query,
            "--clips", str(num_clips),
            "--output", CONFIG["clips_dir"]  # Save to output_videos folder
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            # Run the agent app
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output to find the output clip paths
            output_paths = []
            for line in result.stdout.split('\n'):
                if "Clip" in line and ":" in line and CONFIG["clips_dir"] in line:
                    path = line.split(":", 1)[1].strip()
                    output_paths.append(path)
            
            self.log(f"Successfully extracted {len(output_paths)} clips")
            print(f"Successfully extracted {len(output_paths)} clips to {CONFIG['clips_dir']}")
            
            return {
                "success": True,
                "output_paths": output_paths,
                "video_filename": video_filename,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            self.log(f"Error running agent app: {e}")
            print(f"Error extracting clips: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def enhance_clips(self, clips_paths: List[str] = None, aspect_ratio: str = "9:16", 
                     quality: str = "720p", caption_style: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process clips with caption_enhancer.py to add captions and resize
        """
        # If no clips provided, use all clips in output_videos directory
        if not clips_paths:
            clips_dir = CONFIG["clips_dir"]  # This should be "output_videos"
            if not os.path.exists(clips_dir):
                return {"error": f"Clips directory not found: {clips_dir}"}
            
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
            clips_paths = []
            
            for filename in os.listdir(clips_dir):
                if filename.lower().endswith(video_extensions):
                    clips_paths.append(os.path.join(clips_dir, filename))
            
            if not clips_paths:
                return {"error": f"No clips found in {clips_dir}"}
        
        # Process clips one by one
        results = []
        for clip_path in clips_paths:
            result = self._enhance_single_clip(clip_path, aspect_ratio, quality, caption_style)
            results.append(result)
        
        successful = sum(1 for r in results if r.get("success", False))
        self.log(f"Enhanced {successful} of {len(results)} clips")
        print(f"Enhanced {successful} of {len(results)} clips. Final clips saved to {CONFIG['output_dir']}")
        
        return {
            "success": True,
            "message": f"Enhanced {successful} of {len(results)} clips",
            "results": results
        }
    
    def _enhance_single_clip(self, clip_path: str, aspect_ratio: str = "9:16", 
                           quality: str = "720p", caption_style: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single clip with caption_enhancer.py
        """
        self.log(f"Enhancing clip: {clip_path}")
        print(f"Enhancing clip: {os.path.basename(clip_path)}")
        
        # Check that the clip exists
        if not os.path.exists(clip_path):
            return {"error": f"Clip file not found: {clip_path}"}
        
        # Prepare command
        cmd = [
            sys.executable,
            "caption_enhancer.py",
            "-i", clip_path,
            "-a", aspect_ratio,
            "-q", quality,
            "-o", CONFIG["output_dir"],  # Save to output_clips folder
            "--non-interactive"  # Add non-interactive mode
        ]
        
        # Add caption style if provided
        if caption_style:
            cmd.extend(["-s", json.dumps(caption_style)])
        
        try:
            # Run the caption enhancer
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output to find the output video path
            output_path = None
            for line in result.stdout.split('\n'):
                if "Output saved to:" in line:
                    output_path = line.split("Output saved to:")[1].strip()
                    break
            
            return {
                "success": True,
                "clip_path": clip_path,
                "output_path": output_path,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            self.log(f"Error running caption enhancer: {e}")
            print(f"Error enhancing clip {os.path.basename(clip_path)}: {e}")
            return {
                "success": False,
                "clip_path": clip_path,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def process_video_workflow(self, video_path: str = None, query: str = "Create engaging clips",
                              num_clips: int = 3, aspect_ratio: str = "9:16", 
                              quality: str = "720p") -> Dict[str, Any]:
        """
        Run the complete workflow: extract clips from a video, then enhance each clip with captions
        """
        try:
            self.log(f"Running complete workflow for video: {video_path}")
            print("=" * 60)
            print("Starting video processing workflow")
            print("=" * 60)
            
            # Step 1: List and select videos
            print("\n--- STEP 1: Video Selection ---")
            videos = self.search_videos("input_videos")
            if videos.get("error"):
                print(f"Error: {videos['error']}")
                return {"success": False, "error": videos["error"]}
                
            if videos.get("count", 0) == 0:
                print("No videos found in input_videos directory.")
                return {"success": False, "error": "No videos found in input_videos directory"}
                
            print(f"\nFound {videos['count']} videos:")
            for i, video in enumerate(videos["videos"]):
                print(f"{i+1}. {video['filename']} ({video['size_mb']} MB)")
                
            while True:
                try:
                    choice = input("\nSelect a video number (or 'all' to process all): ").strip()
                    if choice.lower() == 'all':
                        selected_videos = videos["videos"]
                        break
                    else:
                        choice = int(choice)
                        if 1 <= choice <= len(videos["videos"]):
                            selected_videos = [videos["videos"][choice-1]]
                            break
                        else:
                            print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'all'.")
            
            # Step 2: Extract clips from selected videos
            print("\n--- STEP 2: Extracting clips ---")
            all_extract_results = []
            for video in selected_videos:
                try:
                    print(f"\nProcessing: {video['filename']}")
                    print(f"Video path: {video['path']}")
                    
                    # Verify video exists before processing
                    if not os.path.exists(video['path']):
                        print(f"Error: Video file not found: {video['path']}")
                        continue
                        
                    extract_result = self.extract_clips(video["path"], query, num_clips)
                    all_extract_results.append(extract_result)
                    
                    if not extract_result.get("success", False):
                        error_msg = extract_result.get('error', 'Unknown error')
                        print(f"\nERROR: Failed to extract clips from {video['filename']}: {error_msg}")
                        continue
                        
                    clips_paths = extract_result.get("output_paths", [])
                    if not clips_paths:
                        print(f"\nWARNING: No clips were extracted from {video['filename']}")
                        continue
                        
                    print(f"\nSuccessfully extracted {len(clips_paths)} clips from {video['filename']}")
                    for i, clip_path in enumerate(clips_paths):
                        print(f"{i+1}. {os.path.basename(clip_path)}")
                        
                    # Wait for user verification with timeout
                    print("\nYou have 30 seconds to verify the clips. Press Enter to continue or wait for timeout...")
                    try:
                        verify = input("\nWould you like to verify these clips? (y/n): ").strip().lower()
                    except KeyboardInterrupt:
                        print("\nSkipping verification due to timeout or interruption")
                        verify = 'n'
                        
                    if verify == 'y':
                        print("\nPlease check the clips in the output_videos directory.")
                        try:
                            input("Press Enter when you're ready to continue...")
                        except KeyboardInterrupt:
                            print("\nSkipping clip verification")
                            
                        # Ask if user wants to modify any clips
                        try:
                            modify = input("\nWould you like to modify any clips? (y/n): ").strip().lower()
                        except KeyboardInterrupt:
                            print("\nSkipping clip modification")
                            modify = 'n'
                            
                        if modify == 'y':
                            print("\nPlease make your modifications in the output_videos directory.")
                            try:
                                input("Press Enter when you're done modifying the clips...")
                            except KeyboardInterrupt:
                                print("\nSkipping clip modification")
                                
                except Exception as e:
                    print(f"Error processing video {video['filename']}: {str(e)}")
                    continue
            
            # Step 3: Transcription and enhancement
            print("\n--- STEP 3: Transcription and Enhancement ---")
            for extract_result in all_extract_results:
                try:
                    if not extract_result.get("success", False):
                        continue
                        
                    clips_paths = extract_result.get("output_paths", [])
                    if not clips_paths:
                        continue
                        
                    print(f"\nProcessing {len(clips_paths)} clips from {extract_result['video_filename']}")
                    
                    # Ask for transcription details with timeout
                    try:
                        source_language = input("\nEnter source language code (default: en-US): ").strip() or "en-US"
                    except KeyboardInterrupt:
                        print("\nUsing default language: en-US")
                        source_language = "en-US"
                    
                    # Ask if user wants to verify transcription with timeout
                    try:
                        verify_transcript = input("\nWould you like to verify the transcription? (y/n): ").strip().lower()
                    except KeyboardInterrupt:
                        print("\nSkipping transcription verification")
                        verify_transcript = 'n'
                        
                    if verify_transcript == 'y':
                        print("\nPlease check the transcription in the output_videos directory.")
                        try:
                            input("Press Enter when you're ready to continue...")
                        except KeyboardInterrupt:
                            print("\nSkipping transcription verification")
                            
                        # Ask if user wants to modify transcription with timeout
                        try:
                            modify_transcript = input("\nWould you like to modify the transcription? (y/n): ").strip().lower()
                        except KeyboardInterrupt:
                            print("\nSkipping transcription modification")
                            modify_transcript = 'n'
                            
                        if modify_transcript == 'y':
                            print("\nPlease make your modifications in the output_videos directory.")
                            try:
                                input("Press Enter when you're done modifying the transcription...")
                            except KeyboardInterrupt:
                                print("\nSkipping transcription modification")
                    
                    # Step 4: Enhance clips
                    print(f"\n--- STEP 4: Enhancing clips from {extract_result['video_filename']} ---")
                    try:
                        enhance_result = self.enhance_clips(clips_paths, aspect_ratio, quality)
                        
                        if not enhance_result.get("success", False):
                            print(f"\nERROR: Failed to enhance clips from {extract_result['video_filename']}")
                            continue
                            
                        print(f"\nSuccessfully enhanced {len(enhance_result.get('results', []))} clips")
                        print(f"Final clips saved to: {CONFIG['output_dir']}")
                    except Exception as e:
                        print(f"Error enhancing clips: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Error in transcription/enhancement step: {str(e)}")
                    continue
            
            print("\n--- WORKFLOW COMPLETE ---")
            print("=" * 60)
            
            return {
                "success": True,
                "message": "Completed workflow with user verification steps",
                "extract_results": all_extract_results
            }
            
        except Exception as e:
            print(f"Fatal error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
        
    def search_videos(self, directory: str = None) -> Dict[str, Any]:
        """
        Search for videos in the specified directory
        """
        if not directory:
            directory = "input_videos"  # Default to input_videos, not CONFIG["input_dir"]
            
        self.log(f"Searching for videos in {directory}")
        
        if not os.path.exists(directory):
            return {"error": f"Directory not found: {directory}"}
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        videos = []
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(video_extensions):
                file_path = os.path.join(directory, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                videos.append({
                    "filename": filename,
                    "path": file_path,
                    "size_mb": round(file_size, 2)
                })
        
        return {
            "videos": videos,
            "count": len(videos),
            "directory": directory
        }
    
    def log(self, message: str):
        """Log a message to the log file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Write to log file
        with open(CONFIG["log_file"], "a") as f:
            f.write(log_message + "\n")

def main():
    """Main function for running the video processing workflow"""
    parser = argparse.ArgumentParser(description="Video Processing Workflow Driver")
    parser.add_argument("--query", "-q", help="Query to process clips (prompt for clipping)")
    parser.add_argument("--input", "-i", help="Input video path")
    parser.add_argument("--aspect", "-a", default="9:16", choices=["1:1", "4:5", "9:16", "16:9"], 
                       help="Aspect ratio for video processing")
    parser.add_argument("--quality", default="720p", choices=["480p", "720p", "1080p"], 
                       help="Output quality")
    parser.add_argument("--clips", "-c", type=int, default=3, help="Number of clips to extract")
    parser.add_argument("--extract-only", action="store_true", help="Only extract clips, don't enhance them")
    parser.add_argument("--enhance-only", action="store_true", help="Only enhance existing clips in output_videos")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all videos in input_videos directory")
    
    args = parser.parse_args()
    
    driver = WorkflowDriver()
    
    # Direct command execution based on arguments
    if args.extract_only and args.enhance_only:
        print("Error: Cannot use both --extract-only and --enhance-only flags")
        return
    
    if args.extract_only:
        print("Mode: Extract clips only (input_videos → output_videos)")
        if args.batch:
            # Process all videos in input directory
            result = driver.extract_clips(query=args.query or "Create engaging clips", num_clips=args.clips)
        elif args.input:
            # Process specific video
            result = driver.extract_clips(args.input, args.query or "Create engaging clips", args.clips)
        else:
            print("Error: Must specify --input or --batch with --extract-only")
            return
        return
    
    if args.enhance_only:
        print("Mode: Enhance clips only (output_videos → output_clips)")
        # Enhance all clips in output_videos
        result = driver.enhance_clips(aspect_ratio=args.aspect, quality=args.quality)
        return
    
    # Run full workflow (default)
    if args.input or args.batch:
        print("Mode: Complete workflow (input_videos → output_videos → output_clips)")
        if args.batch:
            driver.process_video_workflow(
                video_path=None,  # Process all videos
                query=args.query or "Create engaging clips", 
                num_clips=args.clips,
                aspect_ratio=args.aspect, 
                quality=args.quality
            )
        else:
            driver.process_video_workflow(
                video_path=args.input,
                query=args.query or "Create engaging clips", 
                num_clips=args.clips,
                aspect_ratio=args.aspect, 
                quality=args.quality
            )
    else:
        # Show interactive menu
        print("=" * 60)
        print("Video Processing Workflow")
        print("=" * 60)
        print("Standard workflow:")
        print("1. Upload videos to input_videos folder")
        print("2. Extract clips with agent_app.py (saved to output_videos)")
        print("3. Enhance clips with caption_enhancer.py (saved to output_clips)")
        print("=" * 60)
        
        videos = driver.search_videos("input_videos")
        if videos.get("count", 0) == 0:
            print("No videos found in input_videos directory.")
            print("Please add videos to the input_videos directory and try again.")
            return
        
        print(f"Found {videos['count']} videos in input_videos directory.")
        
        while True:
            print("\nChoose an option:")
            print("1. Run complete workflow (input_videos → output_videos → output_clips)")
            print("2. Extract clips only (input_videos → output_videos)")
            print("3. Enhance existing clips only (output_videos → output_clips)")
            print("4. Exit")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                query = input("Enter prompt for clip extraction (default: 'Create engaging clips'): ").strip()
                if not query:
                    query = "Create engaging clips"
                
                num_clips = input("Enter number of clips to extract (default: 3): ").strip()
                try:
                    num_clips = int(num_clips) if num_clips else 3
                except ValueError:
                    num_clips = 3
                
                aspect_ratio = input("Enter aspect ratio (1:1, 4:5, 9:16, 16:9) (default: 9:16): ").strip()
                if aspect_ratio not in ["1:1", "4:5", "9:16", "16:9"]:
                    aspect_ratio = "9:16"
                
                quality = input("Enter quality (480p, 720p, 1080p) (default: 720p): ").strip()
                if quality not in ["480p", "720p", "1080p"]:
                    quality = "720p"
                
                driver.process_video_workflow(
                    query=query,
                    num_clips=num_clips,
                    aspect_ratio=aspect_ratio,
                    quality=quality
                )
                
            elif choice == "2":
                query = input("Enter prompt for clip extraction (default: 'Create engaging clips'): ").strip()
                if not query:
                    query = "Create engaging clips"
                
                num_clips = input("Enter number of clips to extract (default: 3): ").strip()
                try:
                    num_clips = int(num_clips) if num_clips else 3
                except ValueError:
                    num_clips = 3
                
                driver.extract_clips(query=query, num_clips=num_clips)
                
            elif choice == "3":
                aspect_ratio = input("Enter aspect ratio (1:1, 4:5, 9:16, 16:9) (default: 9:16): ").strip()
                if aspect_ratio not in ["1:1", "4:5", "9:16", "16:9"]:
                    aspect_ratio = "9:16"
                
                quality = input("Enter quality (480p, 720p, 1080p) (default: 720p): ").strip()
                if quality not in ["480p", "720p", "1080p"]:
                    quality = "720p"
                
                driver.enhance_clips(aspect_ratio=aspect_ratio, quality=quality)
                
            elif choice == "4":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 