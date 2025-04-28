import os
import sys
import subprocess
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import json
import argparse
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import tempfile
import re
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import datetime

# NVIDIA API key
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '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')

# Aspect ratio configurations
ASPECT_RATIOS = {
    "1:1": {"width_ratio": 1, "height_ratio": 1, "description": "Square (Instagram)"},
    "4:5": {"width_ratio": 4, "height_ratio": 5, "description": "Portrait (Instagram)"},
    "9:16": {"width_ratio": 9, "height_ratio": 16, "description": "Vertical (TikTok, Reels)"},
    "16:9": {"width_ratio": 16, "height_ratio": 9, "description": "Landscape (YouTube)"}
}

# Quality level configurations
QUALITY_LEVELS = {
    "480p": {"height": 480, "description": "Standard Definition"},
    "720p": {"height": 720, "description": "HD"},
    "1080p": {"height": 1080, "description": "Full HD"}
}

# Default caption style
DEFAULT_CAPTION_STYLE = {
    "font": "Arial",
    "fontsize": 30,
    "color": "white",
    "highlight_color": "yellow",
    "bg_color": "black",
    "bg_opacity": 0.5,
    "position": "bottom",
    "margin": 50  # Margin from the bottom in pixels
}

# Initialize the LLM
try:
    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct", 
        api_key=NVIDIA_API_KEY,
        streaming=False
    )
    print("Successfully initialized NVIDIA AI")
except Exception as e:
    print(f"Error initializing NVIDIA AI: {e}")
    print("Using fallback mechanism for transcript analysis")
    llm = None

class CaptionEnhancer:
    """Main class for handling video captioning, resizing, and enhancement"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory at: {self.temp_dir}")
        self.input_videos = []
        self.verify_ffmpeg()
        self.check_imagemagick()
    
    def __del__(self):
        """Clean up temporary files"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    def verify_ffmpeg(self):
        """Verify FFmpeg is installed and accessible"""
        # First check for environment variable
        ffmpeg_env_path = os.environ.get('FFMPEG_PATH')
        
        if not ffmpeg_env_path or not os.path.exists(ffmpeg_env_path):
            # Set the path to ffmpeg in the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check several possible locations for FFmpeg
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
            ffmpeg_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            
            if ffmpeg_path:
                ffmpeg_env_path = ffmpeg_path
                os.environ['FFMPEG_PATH'] = ffmpeg_path
                print(f"Found FFmpeg at: {ffmpeg_path}")
            else:
                print("FFmpeg not found. Please set FFMPEG_PATH environment variable.")
                sys.exit(1)
        
        # Add ffmpeg directory to PATH for other tools
        ffmpeg_dir = os.path.dirname(ffmpeg_env_path)
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    def check_imagemagick(self):
        """Check if ImageMagick is installed and available"""
        try:
            from moviepy.config import get_setting, change_settings
            imagemagick_path = get_setting("IMAGEMAGICK_BINARY")
            
            # Print current setting for debugging
            print(f"Current ImageMagick path: {imagemagick_path}")
            
            # Check if ImageMagick is installed but path is not configured properly
            if imagemagick_path is None or imagemagick_path == "unset" or not os.path.exists(imagemagick_path):
                print("ImageMagick not properly configured, searching for installation...")
                
                # Try to find ImageMagick in common locations
                common_paths = [
                    # Windows paths - magick.exe first (newer versions)
                    "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
                    "C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\magick.exe",
                    "C:\\Program Files\\ImageMagick-7.0.11-Q16-HDRI\\magick.exe",
                    "C:\\Program Files\\ImageMagick-7.0.11-Q16\\magick.exe",
                    "C:\\Program Files (x86)\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
                    "C:\\Program Files (x86)\\ImageMagick-7.1.0-Q16-HDRI\\magick.exe",
                    "C:\\Program Files (x86)\\ImageMagick-7.0.11-Q16-HDRI\\magick.exe",
                    "C:\\Program Files (x86)\\ImageMagick-7.0.11-Q16\\magick.exe",
                    # Additional legacy paths for convert.exe
                    "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe",
                    "C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\convert.exe",
                    "C:\\Program Files\\ImageMagick-7.0.11-Q16-HDRI\\convert.exe",
                    "C:\\Program Files\\ImageMagick-7.0.11-Q16\\convert.exe",
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        print(f"Found ImageMagick at {path}")
                        # Update MoviePy settings
                        change_settings({"IMAGEMAGICK_BINARY": path})
                        return True
                
                # Try alternative approach using PATH
                try:
                    # Check if 'magick' is in PATH
                    which_cmd = "where" if os.name == "nt" else "which"
                    for cmd in ["magick", "convert"]:
                        try:
                            result = subprocess.run([which_cmd, cmd], 
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE,
                                                  text=True)
                            if result.returncode == 0 and result.stdout.strip():
                                magick_path = result.stdout.strip().split('\n')[0]
                                # Skip the windows built-in convert.exe
                                if cmd == "convert" and "System32" in magick_path:
                                    print(f"Found Windows System convert.exe (not ImageMagick) at: {magick_path}")
                                    continue
                                    
                                print(f"Found ImageMagick in PATH: {magick_path}")
                                change_settings({"IMAGEMAGICK_BINARY": magick_path})
                                return True
                        except:
                            pass
                except:
                    pass
                
                print("ImageMagick is not installed or not found in common locations.")
                print("To enable text overlays, please install ImageMagick from:")
                print("https://imagemagick.org/script/download.php")
                print("Make sure to check 'Install legacy utilities (convert)' and 'Add to system path' during installation.")
                
                return False
                
            return True
        except Exception as e:
            print(f"Error checking ImageMagick: {e}")
            return False
    
    def find_videos(self, input_path):
        """Find videos at the specified path or pattern"""
        self.input_videos = []
        
        # Ensure output directories exist
        output_videos_dir = "output_videos"
        output_clips_dir = "output_clips"
        
        for dir_path in [output_videos_dir, output_clips_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
        
        # Check if input is a directory
        if os.path.isdir(input_path):
            print(f"Searching for videos in: {input_path}")
            for root, _, files in os.walk(input_path):
                for file in files:
                    if self._is_video_file(file):
                        full_path = os.path.join(root, file)
                        self.input_videos.append(full_path)
        
        # Check if input is a single file
        elif os.path.isfile(input_path) and self._is_video_file(input_path):
            self.input_videos.append(os.path.abspath(input_path))
        
        # Check output directory for existing processed videos
        elif input_path == "output_videos":
            print(f"Searching for videos in: {output_videos_dir}")
            output_videos = []
            for file in os.listdir(output_videos_dir):
                if self._is_video_file(file):
                    full_path = os.path.join(output_videos_dir, file)
                    output_videos.append(full_path)
            self.input_videos = output_videos
        # Check clips directory for existing processed clips
        elif input_path == "output_clips":
            print(f"Searching for videos in: {output_clips_dir}")
            output_clips = []
            for file in os.listdir(output_clips_dir):
                if self._is_video_file(file):
                    full_path = os.path.join(output_clips_dir, file)
                    output_clips.append(full_path)
            self.input_videos = output_clips
        # Special case to search in both output directories
        elif input_path == "all_outputs":
            print("Searching for videos in both output directories...")
            all_outputs = []
            for dir_path in [output_videos_dir, output_clips_dir]:
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if self._is_video_file(file):
                            full_path = os.path.join(dir_path, file)
                            all_outputs.append(full_path)
            self.input_videos = all_outputs
        else:
            # Try to interpret as a glob pattern
            import glob
            for file in glob.glob(input_path):
                if os.path.isfile(file) and self._is_video_file(file):
                    self.input_videos.append(os.path.abspath(file))
        
        return len(self.input_videos) > 0
    
    def _is_video_file(self, file_path):
        """Check if file is a video based on extension"""
        video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
        return any(file_path.lower().endswith(ext) for ext in video_exts)
    
    def list_videos(self):
        """List all found videos with their file sizes"""
        if not self.input_videos:
            print("No videos found.")
            return
        
        print(f"\nFound {len(self.input_videos)} video(s):")
        for i, video_path in enumerate(self.input_videos):
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB
            print(f"{i+1}. {os.path.basename(video_path)} ({file_size:.1f} MB)")
    
    def get_video_info(self, video_path):
        """Get video information using ffmpeg"""
        try:
            ffmpeg_path = os.environ.get('FFMPEG_PATH')
            # Use subprocess with UTF-8 encoding explicitly
            result = subprocess.run(
                [ffmpeg_path, "-i", video_path],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,  # Use text mode
                encoding='utf-8',  # Explicitly use UTF-8
                errors='replace'  # Replace invalid characters
            )
            
            info = {}
            
            # Extract duration
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', result.stderr)
            if duration_match:
                h, m, s = duration_match.groups()
                total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                info['duration'] = total_seconds
                
                # Format for display
                if total_seconds >= 3600:
                    formatted_duration = f"{int(total_seconds // 3600)}h {int((total_seconds % 3600) // 60)}m {int(total_seconds % 60)}s"
                elif total_seconds >= 60:
                    formatted_duration = f"{int(total_seconds // 60)}m {int(total_seconds % 60)}s"
                else:
                    formatted_duration = f"{int(total_seconds)}s"
                
                info['formatted_duration'] = formatted_duration
            
            # Extract resolution
            resolution_match = re.search(r'(\d{2,5})x(\d{2,5})', result.stderr)
            if resolution_match:
                width, height = resolution_match.groups()
                info['width'] = int(width)
                info['height'] = int(height)
                info['resolution'] = f"{width}x{height}"
            
            return info
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None
    
    def transcribe_video(self, video_path, model_name="base"):
        """Transcribe video audio using Whisper"""
        ffmpeg_path = os.environ.get('FFMPEG_PATH')
        audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        # Process the video to wav for Whisper
        try:
            print(f"Extracting audio using FFmpeg from: {video_path}")
            subprocess_cmd = [ffmpeg_path, "-i", video_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-f", "wav", audio_path]
            subprocess.run(subprocess_cmd, check=True)
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
        
        if not os.path.exists(audio_path):
            print("Failed to create audio file")
            return None
            
        print(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name)
        
        print("Starting transcription...")
        
        try:
            # Use more accurate settings while still keeping it fast
            result = model.transcribe(
                audio_path, 
                fp16=False,  # Better accuracy on CPU
                language="en",  # Force English
                task="transcribe"  # Specifically transcribe (not translate)
            )
            
            transcription = []
            for segment in result['segments']:
                # Clean up the text
                text = segment['text'].strip()
                
                # Only add segments with real text content
                if text and len(text) > 1:  # Ensure there's actual content
                    transcription.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text
                    })
            
            # Save transcript to JSON for possible reuse
            transcript_path = os.path.splitext(video_path)[0] + "_transcript.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2)
            
            print(f"Saved transcript to: {transcript_path}")
            
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def detect_subject_position(self, frame):
        """Detect the main subject position in a frame using OpenCV face detection"""
        try:
            import cv2
            import numpy as np
            
            # Convert frame to grayscale for face detection
            # First ensure the frame is in the right format for OpenCV
            if frame.shape[2] == 4:  # RGBA
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Try multiple face detection methods for better results
            
            # 1. First try Haar cascade face detector (fastest but less accurate)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # If Haar cascade found faces, use the largest one
            if len(faces) > 0:
                # Find the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Return the center of the face
                center_x = x + w/2
                center_y = y + h/2
                print(f"Face detected with Haar cascade at ({center_x:.1f}, {center_y:.1f})")
                return center_x, center_y
            
            # 2. If Haar cascade fails, try to use DNN-based face detector if available
            try:
                # Try to load DNN face detector model if it exists
                prototxt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.prototxt")
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res10_300x300_ssd_iter_140000.caffemodel")
                
                # Only proceed if the model files exist
                if os.path.exists(prototxt_path) and os.path.exists(model_path):
                    print("Using DNN face detector")
                    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                    
                    # Get frame dimensions
                    h, w = frame_bgr.shape[:2]
                    
                    # Create a blob from the image
                    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    
                    # Pass the blob through the network
                    face_net.setInput(blob)
                    detections = face_net.forward()
                    
                    # Find the detection with highest confidence
                    max_confidence = 0
                    face_location = None
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5 and confidence > max_confidence:  # Threshold of 0.5
                            max_confidence = confidence
                            
                            # Get the box coordinates
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            x1, y1, x2, y2 = box.astype("int")
                            face_location = (x1, y1, x2-x1, y2-y1)  # x, y, width, height
                    
                    if face_location:
                        x, y, w, h = face_location
                        center_x = x + w/2
                        center_y = y + h/2
                        print(f"Face detected with DNN at ({center_x:.1f}, {center_y:.1f}) with confidence {max_confidence:.2f}")
                        return center_x, center_y
            except Exception as e:
                print(f"DNN face detection failed: {e}")
                # Continue to other methods
            
            # 3. Try to use MediaPipe face detection if available
            try:
                import mediapipe as mp
                
                print("Using MediaPipe face detection")
                mp_face_detection = mp.solutions.face_detection
                
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                    results = face_detection.process(frame_rgb)
                    
                    if results.detections:
                        # Get the first detection
                        detection = results.detections[0]
                        
                        # MediaPipe returns normalized coordinates
                        bbox = detection.location_data.relative_bounding_box
                        ih, iw = frame.shape[:2]
                        
                        # Convert to pixel coordinates
                        x = int(bbox.xmin * iw)
                        y = int(bbox.ymin * ih)
                        w = int(bbox.width * iw)
                        h = int(bbox.height * ih)
                        
                        center_x = x + w/2
                        center_y = y + h/2
                        print(f"Face detected with MediaPipe at ({center_x:.1f}, {center_y:.1f})")
                        return center_x, center_y
            except ImportError:
                print("MediaPipe not available")
            except Exception as e:
                print(f"MediaPipe face detection failed: {e}")
            
            # 4. If face detection fails, try to detect other objects in the scene
            print("No faces found, attempting to detect other objects...")
            
            # Create HOG descriptor for people detection
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                
                # Detect people
                people, _ = hog.detectMultiScale(
                    frame_bgr,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=1.05
                )
                
                if len(people) > 0:
                    # Find the largest person
                    largest_person = max(people, key=lambda person: person[2] * person[3])
                    x, y, w, h = largest_person
                    
                    center_x = x + w/2
                    center_y = y + h/2
                    print(f"Person detected at ({center_x:.1f}, {center_y:.1f})")
                    return center_x, center_y
            except Exception as e:
                print(f"HOG people detection failed: {e}")
            
            # 5. Last resort: look for prominent edges and contours
            try:
                # Use Canny edge detection with more appropriate thresholds
                edges = cv2.Canny(gray, 100, 200)
                
                # Dilate edges to connect nearby contours
                kernel = np.ones((5,5), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Find contours
                contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Filter out small contours
                    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
                    
                    if significant_contours:
                        # Get the contour with largest area
                        largest_contour = max(significant_contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        
                        # Calculate center of mass if moments are valid
                        if M["m00"] > 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            print(f"Object detected at ({center_x:.1f}, {center_y:.1f})")
                            return center_x, center_y
            except Exception as e:
                print(f"Contour detection failed: {e}")
            
            # Default: Return center of frame if no subjects found
            height, width = frame.shape[:2]
            print("No subjects detected, using center of frame")
            return width / 2, height / 2
            
        except Exception as e:
            print(f"Error in subject detection: {e}")
            # Return the center of the frame as fallback
            height, width = frame.shape[:2]
            return width / 2, height / 2
    
    def auto_reframe(self, video_clip, target_aspect_ratio):
        """Intelligently reframes the video based on subject detection"""
        try:
            # Get original dimensions
            orig_width, orig_height = video_clip.size
            orig_aspect = orig_width / orig_height
            
            # Calculate target aspect ratio
            target_aspect = (ASPECT_RATIOS[target_aspect_ratio]["width_ratio"] / 
                            ASPECT_RATIOS[target_aspect_ratio]["height_ratio"])
            
            # Skip reframing if aspect ratios are already similar
            if abs(orig_aspect - target_aspect) < 0.1:
                print(f"Aspect ratios are similar ({orig_aspect:.2f} vs {target_aspect:.2f}), no need to crop")
                return video_clip.resize(width=orig_width, height=orig_height)
            
            # Calculate dimensions of the crop window
            if orig_aspect > target_aspect:
                # Original is wider, need to crop width
                new_width = int(orig_height * target_aspect)
                # Ensure new_width is even
                new_width = new_width + (new_width % 2)
                crop_height = orig_height
            else:
                # Original is taller, need to crop height
                new_width = orig_width
                crop_height = int(orig_width / target_aspect)
                # Ensure crop_height is even
                crop_height = crop_height + (crop_height % 2)
            
            # Sample multiple frames for subject detection for better results
            print("Analyzing video for intelligent reframing...")
            
            # List to store subject positions
            subject_positions = []
            
            # Try 3 frames - beginning, middle, and end
            try:
                # Get total duration
                duration = video_clip.duration
                
                # Sample at different timestamps to handle movement
                sample_times = [
                    0,                      # beginning
                    duration * 0.25,        # quarter
                    duration * 0.5,         # middle
                    duration * 0.75,        # three-quarters
                    max(0, duration - 1)    # near end (avoid exact end frame)
                ]
                
                for t in sample_times:
                    try:
                        frame = video_clip.get_frame(t)
                        x, y = self.detect_subject_position(frame)
                        subject_positions.append((x, y))
                        print(f"At {t:.2f}s: detected subject at ({x:.1f}, {y:.1f})")
                    except Exception as e:
                        print(f"Error sampling frame at {t:.2f}s: {e}")
                
                # If no frames were successfully analyzed, fall back to first frame
                if not subject_positions:
                    print("Falling back to first frame analysis")
                    first_frame = video_clip.get_frame(0)
                    x, y = self.detect_subject_position(first_frame)
                    subject_positions.append((x, y))
                
                # Calculate the average position (more stable than just using one frame)
                sum_x, sum_y = 0, 0
                for x, y in subject_positions:
                    sum_x += x
                    sum_y += y
                
                center_x = sum_x / len(subject_positions)
                center_y = sum_y / len(subject_positions)
                print(f"Final subject position estimate: ({center_x:.1f}, {center_y:.1f})")
                
            except Exception as e:
                print(f"Error during video analysis: {e}")
                # Default to center of frame
                center_x, center_y = orig_width / 2, orig_height / 2
            
            # Apply smart zoom by adjusting crop window to keep subject in frame
            zoom_factor = 1.0  # Default: no zoom
            
            # Only apply zoom for vertical videos
            if target_aspect_ratio == "9:16" and orig_aspect < 1.0:
                # For already vertical videos being converted to 9:16, apply moderate zoom
                zoom_factor = 1.1  # 10% zoom
                print(f"Applying zoom factor of {zoom_factor}")
            
            # Keep the subject in frame by adjusting crop window
            if orig_aspect > target_aspect:
                # Cropping width - adjust x position
                half_width = (new_width / zoom_factor) / 2
                min_x = half_width
                max_x = orig_width - half_width
                
                # Bias towards center slightly to avoid extreme crops
                # This weighted average keeps us from going too far to the edges
                center_x = (center_x * 0.7) + (orig_width / 2 * 0.3)
                
                center_x = max(min_x, min(center_x, max_x))
                
                # Calculate crop coordinates
                x1 = max(0, int(center_x - half_width))
                # Ensure x1 is even
                x1 = x1 - (x1 % 2)
                
                # Adjust new_width if zooming
                actual_width = int(new_width / zoom_factor)
                # Ensure even
                actual_width = actual_width + (actual_width % 2)
                
                print(f"Cropping to width {actual_width} at x position {x1}")
                
                # Apply crop
                new_clip = video_clip.crop(x1=x1, y1=0, width=actual_width, height=crop_height)
                
                # If we applied zoom, resize to target dimensions
                if zoom_factor != 1.0:
                    new_clip = new_clip.resize(width=new_width, height=crop_height)
                
            else:
                # Cropping height - adjust y position
                half_height = (crop_height / zoom_factor) / 2
                min_y = half_height
                max_y = orig_height - half_height
                
                # Bias towards center slightly to avoid extreme crops
                # This weighted average keeps us from going too far to the edges
                center_y = (center_y * 0.7) + (orig_height / 2 * 0.3)
                
                center_y = max(min_y, min(center_y, max_y))
                
                # Calculate crop coordinates
                y1 = max(0, int(center_y - half_height))
                # Ensure y1 is even
                y1 = y1 - (y1 % 2)
                
                # Adjust crop_height if zooming
                actual_height = int(crop_height / zoom_factor)
                # Ensure even
                actual_height = actual_height + (actual_height % 2)
                
                print(f"Cropping to height {actual_height} at y position {y1}")
                
                # Apply crop
                new_clip = video_clip.crop(x1=0, y1=y1, width=new_width, height=actual_height)
                
                # If we applied zoom, resize to target dimensions
                if zoom_factor != 1.0:
                    new_clip = new_clip.resize(width=new_width, height=crop_height)
            
            return new_clip
            
        except Exception as e:
            print(f"Auto-reframing failed: {e}")
            print("Falling back to center crop...")
            
            # Use standard center crop as fallback
            if orig_aspect > target_aspect:
                # Original is wider, crop from center
                new_width = int(orig_height * target_aspect)
                # Ensure new_width is even
                new_width = new_width + (new_width % 2)
                x1 = int((orig_width - new_width) / 2)
                # Ensure x1 is even
                x1 = x1 - (x1 % 2)
                return video_clip.crop(x1=x1, y1=0, width=new_width, height=orig_height)
            else:
                # Original is taller, crop from center
                new_height = int(orig_width / target_aspect)
                # Ensure new_height is even
                new_height = new_height + (new_height % 2)
                y1 = int((orig_height - new_height) / 2)
                # Ensure y1 is even
                y1 = y1 - (y1 % 2)
                return video_clip.crop(x1=0, y1=y1, width=orig_width, height=new_height)
    
    def resize_video(self, video_clip, aspect_ratio, quality):
        """Resize video to target aspect ratio and quality"""
        # Get original dimensions
        orig_width, orig_height = video_clip.size
        orig_aspect = orig_width / orig_height
        
        # Get target dimensions
        target_aspect = (ASPECT_RATIOS[aspect_ratio]["width_ratio"] / 
                         ASPECT_RATIOS[aspect_ratio]["height_ratio"])
        target_height = QUALITY_LEVELS[quality]["height"]
        target_width = int(target_height * target_aspect)
        
        # IMPORTANT: Ensure dimensions are even numbers (required for H.264)
        target_width = target_width + (target_width % 2)
        target_height = target_height + (target_height % 2)
        
        print(f"Resizing from {orig_width}x{orig_height} to {target_width}x{target_height}")
        
        # Ask if user wants AI auto-reframing
        use_auto_reframe = True  # Default to true for automatic reframing
        
        # Check if aspects are different enough to need cropping
        if abs(orig_aspect - target_aspect) > 0.1:
            # Create a new clip with the target dimensions
            if use_auto_reframe:
                print("Using AI auto-reframing to position subjects in frame...")
                # Use intelligent cropping
                reframed_clip = self.auto_reframe(video_clip, aspect_ratio)
                resized_clip = reframed_clip.resize((target_width, target_height))
            else:
                # Original is wider, need to crop width or add vertical bars
                # We'll crop for social media to focus on content
                if orig_aspect > target_aspect:
                    new_width = int(orig_height * target_aspect)
                    # Ensure new_width is even
                    new_width = new_width + (new_width % 2)
                    
                    crop_x1 = int((orig_width - new_width) / 2)
                    crop_x2 = crop_x1 + new_width
                    
                    # Crop and resize
                    cropped_clip = video_clip.crop(x1=crop_x1, y1=0, x2=crop_x2, y2=orig_height)
                    resized_clip = cropped_clip.resize((target_width, target_height))
                else:
                    # Original is taller, need to crop height or add horizontal bars
                    new_height = int(orig_width / target_aspect)
                    # Ensure new_height is even
                    new_height = new_height + (new_height % 2)
                    
                    crop_y1 = int((orig_height - new_height) / 2)
                    crop_y2 = crop_y1 + new_height
                    
                    # Crop and resize
                    cropped_clip = video_clip.crop(x1=0, y1=crop_y1, x2=orig_width, y2=crop_y2)
                    resized_clip = cropped_clip.resize((target_width, target_height))
        else:
            # Aspects are similar, just resize
            resized_clip = video_clip.resize((target_width, target_height))
        
        return resized_clip
    
    def extract_clip(self, video_path, start_time, end_time, output_file=None):
        """Extract a clip from the video using FFmpeg"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            return False
            
        # Create output directory - use output_videos for extracted clips
        output_dir = "output_videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created clips directory: {output_dir}")
            
        # Generate output filename if not provided
        if not output_file:
            # Add timestamp to prevent overwriting
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            clean_name = re.sub(r'[^\w\-\.]', '_', base_name)
            output_file = os.path.join(output_dir, f"{clean_name}_clip_{int(start_time)}s_to_{int(end_time)}s_{timestamp}.mp4")
        elif not os.path.isabs(output_file):
            output_file = os.path.join(output_dir, output_file)
            
        print(f"Extracting clip from {start_time}s to {end_time}s...")
        
        try:
            ffmpeg_path = os.environ.get('FFMPEG_PATH')
            # Use more compatible encoding settings
            subprocess.run([
                ffmpeg_path,
                "-i", video_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264",
                "-profile:v", "main",
                "-level", "3.1",
                "-preset", "medium", 
                "-crf", "23",  # Lower is better quality (range 0-51)
                "-c:a", "aac",
                "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # Enables streaming and faster start
                "-strict", "-2",
                output_file
            ], check=True)
            
            # Show the output paths
            abs_output_dir = os.path.abspath(output_dir)
            abs_output_path = os.path.abspath(output_file)
            
            print(f"\nClip extracted to: {abs_output_path}")
            print(f"Output directory: {abs_output_dir}")
            
            # Ask if user wants to open the video or the folder
            try:
                open_choice = input("\nDo you want to: \n1. Open the clip directly\n2. Open containing folder\n3. Do nothing\nEnter choice (1-3): ")
                if open_choice == "1":
                    self.try_play_video(abs_output_path)
                elif open_choice == "2":
                    self.try_open_folder(abs_output_dir)
                else:
                    print(f"You can find the clip at: {abs_output_path}")
            except:
                # If something goes wrong with the prompt, try to open the folder as fallback
                self.try_open_folder(abs_output_dir)
            
            return output_file
        except Exception as e:
            print(f"Error extracting clip: {e}")
            return False
            
    def process_video(self, video_path, aspect_ratio, quality, caption_style=None, output_dir=None):
        """Process a video with captions, aspect ratio change, and quality adjustment"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            return False
            
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = "output_clips"  # Save to output_clips by default
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error creating output directory: {e}")
                # Try to create on desktop as fallback
                try:
                    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                    output_dir = os.path.join(desktop, "clip_output")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(f"Created alternative output directory at: {output_dir}")
                except:
                    print("Failed to create any output directory")
                    return False
            
        # Check disk space (Windows only)
        # This helps avoid "No space left on device" errors
        low_disk_space = False
        output_abs_dir = os.path.abspath(output_dir)
        if os.name == 'nt':  # Windows
            try:
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(output_abs_dir), 
                                                         None, None, ctypes.pointer(free_bytes))
                free_mb = free_bytes.value / (1024 * 1024)
                
                # Check if we have enough space (3x input file size is a safe estimate)
                input_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                needed_mb = input_size_mb * 3
                
                if free_mb < needed_mb:
                    print(f"\nWARNING: Low disk space detected ({free_mb:.1f}MB free, need ~{needed_mb:.1f}MB)")
                    print("Using more efficient encoding to save space")
                    low_disk_space = True
                    
                    # If extremely low space, try to use a different drive
                    if free_mb < input_size_mb:
                        print("Extremely low disk space. Trying alternative locations...")
                        # Try to use a different drive
                        for drive in ['D:', 'E:', 'F:']:
                            try:
                                alt_dir = os.path.join(drive, '\\', 'clip_output')
                                if not os.path.exists(alt_dir):
                                    os.makedirs(alt_dir)
                                free_bytes = ctypes.c_ulonglong(0)
                                ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(drive), 
                                                                        None, None, ctypes.pointer(free_bytes))
                                alt_free_mb = free_bytes.value / (1024 * 1024)
                                if alt_free_mb > needed_mb:
                                    output_dir = alt_dir
                                    print(f"Using alternative output directory with more space: {output_dir}")
                                    break
                            except:
                                continue
            except Exception as e:
                print(f"Note: Could not check disk space: {e}")
        
        # Get video info
        info = self.get_video_info(video_path)
        if not info:
            print("Error: Could not get video information.")
            return False
            
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print(f"Duration: {info.get('formatted_duration', 'Unknown')}")
        print(f"Original resolution: {info.get('resolution', 'Unknown')}")
        print(f"Target aspect ratio: {aspect_ratio} ({ASPECT_RATIOS[aspect_ratio]['description']})")
        print(f"Target quality: {quality} ({QUALITY_LEVELS[quality]['description']})")
        
        # Use default caption style if none provided
        if caption_style is None:
            caption_style = DEFAULT_CAPTION_STYLE
            
        # Load video clip
        try:
            video_clip = VideoFileClip(video_path)
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
        
        # First resize the video to target aspect ratio and quality
        try:
            video_clip = self.resize_video(video_clip, aspect_ratio, quality)
        except Exception as e:
            print(f"Error resizing video: {e}")
            print("Proceeding with original dimensions")
            
        # If caption style is set, transcribe and add captions
        captions_applied = False
        if caption_style:
            # Check if ImageMagick is installed
            imagemagick_available = False
            from moviepy.config import get_setting
            imagemagick_path = get_setting("IMAGEMAGICK_BINARY")
            if imagemagick_path and os.path.exists(imagemagick_path) and "System32" not in imagemagick_path:
                imagemagick_available = True
            
            # Only try captions if ImageMagick is available
            if not imagemagick_available:
                print("\nImageMagick is not properly installed. Skipping captions.")
                print("To enable captions, install ImageMagick from: https://imagemagick.org/script/download.php")
                print("Make sure to check 'Install legacy utilities (convert)' and 'Add to system path' during installation.")
                caption_style = None
            else:
                print("Transcribing video...")
                transcription = self.transcribe_video(video_path)
                
                if transcription:
                    print(f"Transcription complete with {len(transcription)} segments.")
                    # Try to add captions to video
                    try:
                        # Try to apply captions
                        video_with_captions = self.apply_captions_to_video(video_clip, transcription, caption_style)
                        if video_with_captions is not video_clip:  # If captions were applied
                            video_clip = video_with_captions
                            captions_applied = True
                    except Exception as e:
                        print(f"Error applying captions: {e}")
                        print("Proceeding without captions")
                else:
                    print("Transcription failed, proceeding without captions.")
        
        # Generate output filename
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get the base filename without path and extension
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            clean_name = re.sub(r'[^\w\-\.]', '_', base_name)
            
            # Construct output filename with aspect ratio and quality
            output_file = os.path.join(output_dir, f"{clean_name}_{aspect_ratio.replace(':', '_')}"
                                      f"_{quality}_captioned_{timestamp}.mp4")
            
            # Write the output file
            print(f"Writing output to: {output_file}")
            if low_disk_space:
                # More aggressive compression if disk space is low
                video_clip.write_videofile(
                    output_file, 
                    codec="libx264", 
                    audio_codec="aac",
                    preset="fast" if low_disk_space else "medium",
                    bitrate="1000k" if low_disk_space else "2000k",
                    audio_bitrate="96k" if low_disk_space else "128k",
                    threads=max(1, os.cpu_count() - 1),
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "3.1"]
                )
            else:
                # Normal quality
                video_clip.write_videofile(
                    output_file, 
                    codec="libx264", 
                    audio_codec="aac",
                    preset="medium",
                    threads=max(1, os.cpu_count() - 1),
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "3.1"]
                )
                
            print("Video processing complete!\n")
            
            # Show the output paths
            abs_output_path = os.path.abspath(output_file)
            abs_output_dir = os.path.abspath(output_dir)
            
            print(f"Output saved to: {abs_output_path}")
            print(f"Output directory: {abs_output_dir}")
            
            # Ask if user wants to open the video or the folder
            try:
                # Check if we're in non-interactive mode from command-line args
                import sys
                non_interactive = "--non-interactive" in sys.argv
                
                if not non_interactive:
                    open_choice = input("\nDo you want to:\n1. Open the video file directly\n2. Open containing folder\n3. Do nothing\nEnter choice (1-3): ")
                    if open_choice == "1":
                        self.try_play_video(abs_output_path)
                    elif open_choice == "2":
                        self.try_open_folder(abs_output_dir)
                    else:
                        print(f"You can find the video at: {abs_output_path}")
            except:
                # If something goes wrong with the prompt, don't crash
                pass
            
            # Close the clip to release resources
            video_clip.close()
            
            return abs_output_path
        except Exception as e:
            print(f"Error writing video: {e}")
            print("Check disk space and permissions.")
            
            # Try to close the clip
            try:
                video_clip.close()
            except:
                pass
                
            return False
    
    def apply_captions_to_video(self, video_clip, transcription, caption_style):
        """Apply captions with word-level highlighting to a video clip"""
        if not transcription:
            return video_clip
            
        print("Applying captions with word-level highlighting...")
        try:
            # Import all required modules manually
            import moviepy.video.tools.subtitles as subs
            import numpy as np
            from moviepy.video.VideoClip import TextClip, ColorClip 
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            
            # Create a ColorClip for the background
            video_width, video_height = video_clip.size
            
            # Calculate better caption width based on aspect ratio
            # For vertical videos, make captions wider
            aspect_ratio = video_width / video_height
            
            # For vertical videos (aspect ratio < 1), use wider captions
            if aspect_ratio < 1:  # Vertical video
                caption_width_pct = 0.9  # Use 90% of video width
            else:  # Standard or landscape video
                caption_width_pct = 0.8  # Use 80% of video width
                
            max_caption_width = int(video_width * caption_width_pct)
            
            # Function to generate subtitles with word highlighting
            def make_subtitle(txt, start, end, highlight_word=None):
                try:
                    # Ensure text is a string
                    txt = str(txt)
                    
                    # Adjust font size based on video dimensions
                    # For smaller videos, use smaller font
                    adjusted_fontsize = caption_style["fontsize"]
                    if video_width < 640:  # For small videos
                        adjusted_fontsize = max(16, int(adjusted_fontsize * 0.8))
                    elif video_width > 1280:  # For large videos
                        adjusted_fontsize = min(48, int(adjusted_fontsize * 1.2))
                    
                    # Create TextClip with proper width and line breaks
                    txt_clip = TextClip(
                        txt, 
                        fontsize=adjusted_fontsize,
                        color=caption_style["color"],
                        font=caption_style["font"],
                        method='caption',
                        size=(max_caption_width, None),  # Fixed width with auto height
                        align='center'  # Center-align the text
                    )
                    
                    # Get dimensions of the text clip
                    if hasattr(txt_clip, 'w') and txt_clip.w is not None:
                        bg_width = min(max_caption_width, txt_clip.w + 40)  # Add padding
                    else:
                        bg_width = max_caption_width
                        
                    if hasattr(txt_clip, 'h') and txt_clip.h is not None:
                        bg_height = txt_clip.h + 20  # Add padding
                    else:
                        bg_height = 50
                    
                    # Create black background
                    bg_clip = ColorClip(
                        size=(int(bg_width), int(bg_height)),
                        color=(0, 0, 0)  # Use simple RGB tuple instead of string
                    )
                    
                    # Set opacity safely (use set_opacity only on bg_clip)
                    try:
                        bg_clip = bg_clip.set_opacity(0.7)  # Slightly more opaque for readability
                    except Exception as e:
                        print(f"Warning: Could not set background opacity: {e}")
                    
                    # Position text on background 
                    txt_positioned = txt_clip.set_position('center')
                    
                    # Composite the text over the background using CompositeVideoClip
                    final_clip = CompositeVideoClip([bg_clip, txt_positioned])
                    
                    # Position the caption
                    if caption_style["position"] == "top":
                        position = ('center', 50)  # Fixed margin
                    elif caption_style["position"] == "middle":
                        position = ('center', 'center')
                    else:  # bottom (default)
                        position = ('center', video_height - 100)  # Fixed margin
                    
                    # Set position, start and end time
                    final_clip = final_clip.set_position(position).set_start(start).set_end(end)
                    return final_clip
                    
                except Exception as e:
                    print(f"Error in make_subtitle for '{txt}': {e}")
                    return None
            
            # Process each segment of the transcript
            subtitle_clips = []
            
            for segment in transcription:
                try:
                    start_time = float(segment['start'])
                    end_time = float(segment['end'])
                    text = str(segment['text']).strip()
                    
                    # Skip empty segments
                    if not text:
                        continue
                        
                    # Create one subtitle per segment (simpler approach)
                    subtitle_clip = make_subtitle(text, start_time, end_time)
                    if subtitle_clip is not None:
                        subtitle_clips.append(subtitle_clip)
                except Exception as e:
                    print(f"Error creating segment subtitle: {e}")
            
            # Composite all subtitle clips with the video
            if subtitle_clips:
                try:
                    final_video = CompositeVideoClip([video_clip] + subtitle_clips)
                    # Ensure the final video duration matches the original
                    final_video = final_video.set_duration(video_clip.duration)
                    # Copy fps from original
                    if hasattr(video_clip, 'fps') and video_clip.fps:
                        final_video = final_video.set_fps(video_clip.fps)
                    return final_video
                except Exception as e:
                    print(f"Error compositing subtitles with video: {e}")
                    return video_clip
            else:
                print("No valid subtitle clips created")
                return video_clip
            
        except Exception as e:
            print(f"Error applying captions: {e}")
            print("Proceeding without captions")
            return video_clip
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("AI Video Enhancement Processor")
        print("=============================")
        
        # Ask for clip extraction or full processing
        menu_choice = input("Choose an option:\n1. Process a video (resize, add captions)\n2. Extract clips from an existing video\n3. Quick extract (create a simple clip)\nEnter option (1-3, default: 1): ")
        
        # Process based on user choice
        if menu_choice == "2":
            result = self.extract_clips_interactive()
            if result:
                print("\nSuccessfully extracted clips. You can find them in the 'output_videos' directory.")
            return result
        elif menu_choice == "3":
            result = self.quick_extract()
            return result
        else:
            # Default option 1: Process a video
            
            # Look for input videos in either input_videos or output_videos
            if self.find_videos("output_videos"):
                print("Found videos in output_videos directory.")
            elif self.find_videos("input_videos"):
                print("Found videos in input_videos directory.")
            else:
                print("No videos found in either output_videos or input_videos directories.")
                # Create the directories if they don't exist
                for dir_path in ["input_videos", "output_videos", "output_clips"]:
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                        print(f"Created directory: {dir_path}")
                
                print("Please add videos to the 'input_videos' directory or extract clips to 'output_videos'.")
                return False
            
            self.list_videos()
            
            # Select a video
            selected_idx = 0
            if len(self.input_videos) > 1:
                try:
                    selected_idx = int(input(f"Select a video (1-{len(self.input_videos)}): ")) - 1
                    if selected_idx < 0 or selected_idx >= len(self.input_videos):
                        print("Invalid selection. Using first video.")
                        selected_idx = 0
                except ValueError:
                    print("Invalid input. Using first video.")
                    selected_idx = 0
            
            selected_video = self.input_videos[selected_idx]
            print(f"\nSelected video: {os.path.basename(selected_video)}")
            
            # Get video info
            info = self.get_video_info(selected_video)
            if info:
                print(f"Duration: {info.get('formatted_duration', 'Unknown')}")
                print(f"Original resolution: {info.get('resolution', 'Unknown')}")
            
            # Ask for aspect ratio
            print("\nAvailable aspect ratios:")
            for i, (ratio, data) in enumerate(ASPECT_RATIOS.items()):
                print(f"{i+1}. {ratio} - {data['description']}")
            
            try:
                aspect_idx = int(input("Select aspect ratio (default: 3 for 9:16 vertical): ") or "3") - 1
                if aspect_idx < 0 or aspect_idx >= len(ASPECT_RATIOS):
                    print("Invalid selection. Using 9:16 (vertical).")
                    aspect_ratio = "9:16"
                else:
                    aspect_ratio = list(ASPECT_RATIOS.keys())[aspect_idx]
            except ValueError:
                print("Invalid input. Using 9:16 (vertical).")
                aspect_ratio = "9:16"
            
            # Ask for quality
            print("\nAvailable quality levels:")
            for i, (quality, data) in enumerate(QUALITY_LEVELS.items()):
                print(f"{i+1}. {quality} - {data['description']}")
            
            try:
                quality_idx = int(input("Select quality (default: 2 for 720p): ") or "2") - 1
                if quality_idx < 0 or quality_idx >= len(QUALITY_LEVELS):
                    print("Invalid selection. Using 720p.")
                    quality = "720p"
                else:
                    quality = list(QUALITY_LEVELS.keys())[quality_idx]
            except ValueError:
                print("Invalid input. Using 720p.")
                quality = "720p"
            
            # Caption settings
            add_captions = input("\nAdd captions to the video? (y/n, default: y): ").lower() != 'n'
            
            caption_style = None
            if add_captions:
                caption_style = dict(DEFAULT_CAPTION_STYLE)
                
                # Font selection (simplified for now)
                print("\nSelect font:")
                fonts = ["Arial", "Verdana", "Times New Roman", "Courier New", "Comic Sans MS"]
                for i, font in enumerate(fonts):
                    print(f"{i+1}. {font}")
                
                try:
                    font_idx = int(input(f"Select font (default: 1 for {fonts[0]}): ") or "1") - 1
                    if 0 <= font_idx < len(fonts):
                        caption_style["font"] = fonts[font_idx]
                except ValueError:
                    pass  # Keep default
                
                # Font size
                try:
                    size = int(input(f"Font size (default: {caption_style['fontsize']}): ") or f"{caption_style['fontsize']}")
                    if 10 <= size <= 100:
                        caption_style["fontsize"] = size
                except ValueError:
                    pass  # Keep default
                
                # Text colors
                print("\nSelect text color:")
                colors = ["white", "black", "yellow", "red", "blue", "green", "cyan", "magenta"]
                for i, color in enumerate(colors):
                    print(f"{i+1}. {color}")
                
                try:
                    color_idx = int(input(f"Base text color (default: 1 for white): ") or "1") - 1
                    if 0 <= color_idx < len(colors):
                        caption_style["color"] = colors[color_idx]
                        
                    highlight_idx = int(input(f"Highlight text color (default: 3 for yellow): ") or "3") - 1
                    if 0 <= highlight_idx < len(colors):
                        caption_style["highlight_color"] = colors[highlight_idx]
                except ValueError:
                    pass  # Keep default
                
                # Caption position
                print("\nSelect caption position:")
                positions = ["bottom", "middle", "top"]
                for i, pos in enumerate(positions):
                    print(f"{i+1}. {pos}")
                
                try:
                    pos_idx = int(input(f"Position (default: 1 for bottom): ") or "1") - 1
                    if 0 <= pos_idx < len(positions):
                        caption_style["position"] = positions[pos_idx]
                except ValueError:
                    pass  # Keep default
            
            # Confirm settings
            print("\nProcessing with these settings:")
            print(f"Video: {os.path.basename(selected_video)}")
            print(f"Aspect ratio: {aspect_ratio} ({ASPECT_RATIOS[aspect_ratio]['description']})")
            print(f"Quality: {quality} ({QUALITY_LEVELS[quality]['description']})")
            
            if caption_style:
                print("Captions: Enabled")
                print(f"  Font: {caption_style['font']} ({caption_style['fontsize']}px)")
                print(f"  Colors: {caption_style['color']} base, {caption_style['highlight_color']} highlight")
                print(f"  Position: {caption_style['position']}")
            else:
                print("Captions: Disabled")
            
            proceed = input("\nProceed with these settings? (y/n, default: y): ").lower() != 'n'
            
            if not proceed:
                print("Operation cancelled.")
                return False
                
            # Process the video (with output dir set to output_clips)
            result = self.process_video(selected_video, aspect_ratio, quality, caption_style, output_dir="output_clips")
            return bool(result)
    
    def extract_clips_interactive(self):
        """Interactive mode for clip extraction"""
        # Ask for number of clips first
        try:
            num_clips = int(input("\nHow many clips would you like to process? "))
            if num_clips <= 0:
                print("Number of clips must be positive. Using 1.")
                num_clips = 1
        except ValueError:
            print("Invalid input. Using 1 clip.")
            num_clips = 1
        
        # Find processed videos
        video_found = False
        
        # First try to find videos in input_videos directory
        if self.find_videos("input_videos"):
            video_found = True
            print("Found videos in input_videos directory.")
        
        # Then try output_videos as a fallback
        if not video_found and self.find_videos("output_videos"):
            video_found = True
            print("Found videos in output_videos directory.")
        
        if not video_found:
            print("No videos found in input directories.")
            return False
        
        self.list_videos()
        
        clips_extracted = []
        
        # Process each clip
        for i in range(num_clips):
            print(f"\nClip {i+1}/{num_clips}:")
            
            # Select a video for this clip
            selected_idx = 0
            if len(self.input_videos) > 1:
                try:
                    selected_idx = int(input(f"Select a video (1-{len(self.input_videos)}): ")) - 1
                    if selected_idx < 0 or selected_idx >= len(self.input_videos):
                        print("Invalid selection. Using first video.")
                        selected_idx = 0
                except ValueError:
                    print("Invalid input. Using first video.")
                    selected_idx = 0
            
            selected_video = self.input_videos[selected_idx]
            
            # Get video info
            info = self.get_video_info(selected_video)
            if not info:
                print("Error: Could not get video information.")
                continue
            
            duration = info.get('duration', 0)
            
            print(f"\nVideo: {os.path.basename(selected_video)}")
            print(f"Duration: {info.get('formatted_duration', 'Unknown')} ({duration:.2f}s)")
            
            # Ask for custom start and end times
            try:
                use_custom_time = input("Use custom start/end times? (y/n, default: n): ").lower() == 'y'
                
                if use_custom_time:
                    start_time = float(input(f"Start time (0-{duration:.2f}s): ") or "0")
                    end_time = float(input(f"End time ({start_time:.2f}-{duration:.2f}s): ") or str(duration))
                    
                    # Validate times
                    if start_time < 0:
                        start_time = 0
                    if end_time > duration:
                        end_time = duration
                    if start_time >= end_time:
                        print("Start time must be less than end time. Using full clip.")
                        start_time = 0
                        end_time = duration
                else:
                    # Use the full clip
                    start_time = 0
                    end_time = duration
                
                # Extract the clip (output directory is handled in extract_clip)
                result = self.extract_clip(selected_video, start_time, end_time)
                
                if result:
                    clips_extracted.append(result)
                    print(f"✓ Extracted clip {i+1}/{num_clips}")
                else:
                    print(f"✗ Failed to extract clip {i+1}/{num_clips}")
            except Exception as e:
                print(f"Error extracting clip: {e}")
        
        # Summary
        if clips_extracted:
            print(f"\nSuccessfully extracted {len(clips_extracted)}/{num_clips} clips:")
            for clip in clips_extracted:
                print(f"  - {os.path.basename(clip)}")
            print("\nAll clips are saved in the 'output_videos' directory.")
            
            # Try to open the folder for the user
            try:
                output_dir = os.path.abspath("output_videos")
                print(f"\nOpening output folder: {output_dir}")
                if os.name == 'nt':  # Windows
                    os.system(f'explorer "{output_dir}"')
                elif os.name == 'posix':  # macOS or Linux
                    os.system(f'open "{output_dir}"')
            except Exception as e:
                print(f"Could not open output directory automatically: {e}")
                print(f"Please navigate to: {output_dir}")
            
            return True
        else:
            print("\nFailed to extract any clips.")
            return False
            
    def quick_extract(self):
        """Quick clip extraction from a full video"""
        # First check for videos
        if not self.find_videos("input_videos"):
            print("No videos found in input_videos directory.")
            return False
            
        if len(self.input_videos) == 0:
            print("No videos found.")
            return False
            
        # If multiple videos, allow selection
        selected_idx = 0
        if len(self.input_videos) > 1:
            self.list_videos()
            try:
                selected_idx = int(input(f"Select video (1-{len(self.input_videos)}): ")) - 1
                if selected_idx < 0 or selected_idx >= len(self.input_videos):
                    print("Invalid selection. Using first video.")
                    selected_idx = 0
            except ValueError:
                print("Invalid input. Using first video.")
                selected_idx = 0
                
        video_path = self.input_videos[selected_idx]
        print(f"Selected: {os.path.basename(video_path)}")
        
        # Get video info
        info = self.get_video_info(video_path)
        if not info:
            print("Error: Could not get video information.")
            return False
            
        duration = info.get('duration', 0)
        print(f"Duration: {info.get('formatted_duration', 'Unknown')} ({duration:.2f}s)")
        
        # Ask for clip details
        clip_name = input("\nEnter a name for the clip (optional): ")
        
        # Ask for start and end times
        try:
            start_time = float(input(f"Start time (0-{duration:.2f}s): ") or "0")
            end_time = float(input(f"End time ({start_time:.2f}-{duration:.2f}s): ") or str(duration))
            
            if start_time < 0:
                start_time = 0
            if end_time > duration:
                end_time = duration
            if start_time >= end_time:
                print("Start time must be less than end time. Using full clip.")
                start_time = 0
                end_time = duration
        except ValueError:
            print("Invalid input. Using full video.")
            start_time = 0
            end_time = duration
        
        # Generate output filename
        output_file = None
        if clip_name:
            # Clean up the name
            clip_name = re.sub(r'[^\w\-\.]', '_', clip_name)
            output_file = f"{clip_name}_{int(start_time)}s_to_{int(end_time)}s.mp4"
        
        # Extract the clip
        print(f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s...")
        result = self.extract_clip(video_path, start_time, end_time, output_file)
        
        if result:
            print(f"Clip extracted to: {result}")
            
            # Try to open the output folder
            try:
                output_dir = os.path.abspath("output_videos")
                print(f"\nOpening output folder: {output_dir}")
                if os.name == 'nt':  # Windows
                    os.system(f'explorer "{output_dir}"')
                elif os.name == 'posix':  # macOS or Linux
                    os.system(f'open "{output_dir}"')
            except Exception as e:
                print(f"Could not open output directory automatically: {e}")
            
            # Ask if user wants to process the clip further
            process_clip = input("\nDo you want to process this clip (add captions, resize)? (y/n, default: n): ").lower() == 'y'
            if process_clip:
                # Update input_videos with the new clip
                self.input_videos = [result]
                # Process the video with output to output_clips
                caption_style = dict(DEFAULT_CAPTION_STYLE)
                aspect_ratio = "9:16"  # Default to vertical
                quality = "720p"      # Default to HD
                
                print("\nProcessing clip with default settings:")
                print(f"Aspect ratio: {aspect_ratio} ({ASPECT_RATIOS[aspect_ratio]['description']})")
                print(f"Quality: {quality} ({QUALITY_LEVELS[quality]['description']})")
                print("Caption style: Default")
                
                result = self.process_video(result, aspect_ratio, quality, caption_style, output_dir="output_clips")
                return bool(result)
            else:
                return True
        else:
            print("Failed to extract clip.")
            return False

    def main(self):
        """Main execution function"""
        parser = argparse.ArgumentParser(
            description="Caption Enhancer - Add dual-color captions and convert video aspect ratios"
        )
        
        # Main operation modes
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("-i", "--interactive", action="store_true", 
                              help="Run in interactive mode with prompts for all options")
        mode_group.add_argument("-c", "--clip", action="store_true",
                              help="Extract clips from an existing video")
        mode_group.add_argument("--process-clip", action="store_true",
                              help="Create a new clip directly (shortcut mode)")
                              
        # Video input options
        parser.add_argument("input", nargs="?", default="input_videos",
                          help="Video file or directory (default: 'input_videos' directory)")
        
        # Video processing options
        parser.add_argument("-a", "--aspect-ratio", choices=list(ASPECT_RATIOS.keys()), default="9:16",
                          help="Target aspect ratio for the output video (default: 9:16)")
        parser.add_argument("-q", "--quality", choices=list(QUALITY_LEVELS.keys()), default="720p",
                          help="Target quality level for the output (default: 720p)")
        parser.add_argument("-o", "--output-dir", 
                          help="Output directory (default: output_clips for processing, output_videos for clip extraction)")
        parser.add_argument("--non-interactive", action="store_true",
                          help="Run in non-interactive mode (skip all prompts)")
        
        # Caption options
        caption_group = parser.add_argument_group("Caption Options")
        caption_group.add_argument("--no-captions", action="store_true",
                                 help="Disable caption generation")
        caption_group.add_argument("--caption-font", default=DEFAULT_CAPTION_STYLE["font"],
                                 help=f"Font to use for captions (default: {DEFAULT_CAPTION_STYLE['font']})")
        caption_group.add_argument("--caption-size", type=int, default=DEFAULT_CAPTION_STYLE["fontsize"],
                                 help=f"Font size for captions (default: {DEFAULT_CAPTION_STYLE['fontsize']})")
        caption_group.add_argument("--caption-color", default=DEFAULT_CAPTION_STYLE["color"],
                                 help=f"Base text color for captions (default: {DEFAULT_CAPTION_STYLE['color']})")
        caption_group.add_argument("--caption-highlight", default=DEFAULT_CAPTION_STYLE["highlight_color"],
                                 help=f"Highlight color for current word (default: {DEFAULT_CAPTION_STYLE['highlight_color']})")
        caption_group.add_argument("--caption-position", choices=["bottom", "middle", "top"], default="bottom",
                                 help="Position of captions (default: bottom)")
        caption_group.add_argument("-s", "--caption-style", type=str,
                                 help="JSON string with caption style parameters")
        
        # Clip extraction options
        clip_group = parser.add_argument_group("Clip Extraction Options")
        clip_group.add_argument("--start", type=float,
                              help="Start time for clip extraction (in seconds)")
        clip_group.add_argument("--end", type=float,
                              help="End time for clip extraction (in seconds)")
        clip_group.add_argument("--output-clip", type=str,
                              help="Output filename for extracted clip (will be placed in output_videos)")
        
        args = parser.parse_args()
        
        # Set output directory based on operation mode
        if args.output_dir:
            output_dir = args.output_dir
        elif args.clip:
            output_dir = "output_videos"  # Use output_videos for clip extraction
        else:
            output_dir = "output_clips"  # Use output_clips for processing
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Interactive mode (default with -i flag or if no other mode selected)
        if args.interactive:
            return self.interactive_mode()
        
        # Process clip mode (quick extract)
        if args.process_clip:
            return self.quick_extract()
        
        # If start and end times are provided, and input is a file, do direct clip extraction
        if args.start is not None and args.end is not None and os.path.isfile(args.input):
            result = self.extract_clip(args.input, args.start, args.end, args.output_clip)
            if result:
                print(f"Clip successfully extracted to: {result}")
                return 0
            else:
                print("Failed to extract clip")
                return 1
        
        # Clip extraction mode
        if args.clip:
            # Single file direct extraction
            if os.path.isfile(args.input):
                if args.start is not None and args.end is not None:
                    result = self.extract_clip(args.input, args.start, args.end, args.output_clip)
                    return 0 if result else 1
                elif not args.non_interactive:
                    success = self.extract_clips_interactive()
                    return 0 if success else 1
                else:
                    print("Error: In non-interactive mode, you must provide --start and --end times")
                    return 1
            # Interactive extraction
            elif not args.non_interactive:
                if self.find_videos(args.input):
                    success = self.extract_clips_interactive()
                    return 0 if success else 1
                else:
                    print(f"No videos found matching: {args.input}")
                    return 1
            else:
                print("Error: In non-interactive mode with --clip, you must provide a specific input file")
                return 1
        
        # Find videos from input pattern
        if not self.find_videos(args.input):
            print(f"No videos found matching: {args.input}")
            return 1
        
        if not args.non_interactive:
            self.list_videos()
        
        # Parse caption style
        caption_style = None
        if not args.no_captions:
            if args.caption_style:
                try:
                    caption_style = json.loads(args.caption_style)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON for caption style")
                    return 1
            else:
                caption_style = {
                    "font": args.caption_font,
                    "fontsize": args.caption_size,
                    "color": args.caption_color,
                    "highlight_color": args.caption_highlight,
                    "bg_color": DEFAULT_CAPTION_STYLE["bg_color"],
                    "bg_opacity": DEFAULT_CAPTION_STYLE["bg_opacity"],
                    "position": args.caption_position,
                    "margin": DEFAULT_CAPTION_STYLE["margin"]
                }
        
        # Single video mode - just process the first video found
        if len(self.input_videos) >= 1:
            video_to_process = self.input_videos[0]
            result = self.process_video(
                video_to_process,
                args.aspect_ratio,
                args.quality,
                caption_style=caption_style,
                output_dir=output_dir
            )
            
            if result:
                output_path = result
                print("Video processing completed successfully!")
                print(f"You can find the processed video in the '{output_dir}' directory.")
                
                if not args.non_interactive:
                    # Try to open the output folder
                    self.try_open_folder(os.path.abspath(output_dir))
                
                return 0
            else:
                print("Video processing failed.")
                return 1
        else:
            print("No videos found to process.")
            return 1

    def try_open_folder(self, folder_path):
        """Attempt to open a folder using the most reliable method for each OS"""
        try:
            print(f"Attempting to open folder: {folder_path}")
            
            # For Windows systems
            if os.name == 'nt':
                try:
                    # Try a direct subprocess call first (most reliable)
                    import subprocess
                    subprocess.run(['explorer', folder_path], check=False)
                except Exception as e:
                    print(f"Could not open folder with explorer: {e}")
                    # Try the start command as fallback
                    try:
                        os.system(f'start "" "{folder_path}"')
                    except:
                        pass
            
            # For macOS and Linux
            elif os.name == 'posix':
                import platform
                if platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{folder_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{folder_path}"')
                    
        except Exception as e:
            # Don't throw an error, just print a message
            print(f"Note: Could not automatically open output folder: {e}")
            print("Please navigate to the folder manually using File Explorer.")
            
    def try_play_video(self, video_path):
        """Attempt to play a video file using the system's default player"""
        try:
            print(f"Attempting to play video: {video_path}")
            
            # For Windows systems
            if os.name == 'nt':
                try:
                    # Use the default association through os.startfile
                    os.startfile(video_path)
                    print("Started video player")
                except Exception as e:
                    print(f"Could not start video player: {e}")
                    # Try another method
                    try:
                        import subprocess
                        subprocess.run(['start', '', video_path], shell=True, check=False)
                    except:
                        print("Could not play video with shell command")
            
            # For macOS and Linux
            elif os.name == 'posix':
                import platform
                if platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{video_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{video_path}"')
                    
        except Exception as e:
            # Don't throw an error, just print a message
            print(f"Note: Could not automatically play video: {e}")
            print("Please open the video file manually using your preferred video player.")


def main():
    """Main entry point for the caption enhancer"""
    enhancer = CaptionEnhancer()
    
    try:
        return enhancer.main()    
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