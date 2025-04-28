import os
import sys
import subprocess
import whisper
import tempfile
from pathlib import Path
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from subject_tracker import SubjectTracker

# Constants for aspect ratios and resolutions
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "9:16": (9, 16)
}

QUALITY_LEVELS = {
    "1080p": 1080,
    "720p": 720, 
    "480p": 480
}

# Default caption styling
DEFAULT_CAPTION_STYLE = {
    "font": "Arial",
    "fontsize": 30,
    "color": "white",
    "bg_color": "black",
    "bg_opacity": 0.5,
    "position": "bottom"
}

class VideoEnhancer:
    def __init__(self):
        # Find FFmpeg path
        self.ffmpeg_path = self._get_ffmpeg_path()
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or set FFMPEG_PATH environment variable.")
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory at: {self.temp_dir}")
        
        # Initialize Whisper model for transcription
        self.whisper_model = None
        
        # Initialize subject tracker
        self.subject_tracker = None
    
    def _get_ffmpeg_path(self) -> str:
        """Find the FFmpeg executable"""
        # First check for environment variable
        ffmpeg_env_path = os.environ.get('FFMPEG_PATH')
        
        if ffmpeg_env_path and os.path.exists(ffmpeg_env_path):
            return ffmpeg_env_path
        
        # Check several possible locations for FFmpeg
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
        
        return None
    
    def _get_target_dimensions(self, aspect_ratio: str, quality_level: str) -> Tuple[int, int]:
        """Calculate the target dimensions based on aspect ratio and quality"""
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Choose from {list(ASPECT_RATIOS.keys())}")
        
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Invalid quality level: {quality_level}. Choose from {list(QUALITY_LEVELS.keys())}")
        
        # Get the base height from quality level
        base_height = QUALITY_LEVELS[quality_level]
        
        # Calculate dimensions based on aspect ratio
        width_ratio, height_ratio = ASPECT_RATIOS[aspect_ratio]
        
        if width_ratio > height_ratio:  # Horizontal aspect ratio
            height = base_height
            width = int(height * width_ratio / height_ratio)
        else:  # Vertical or square aspect ratio
            width = base_height
            height = int(width * height_ratio / width_ratio)
            # Adjust if needed for specific cases
            if aspect_ratio == "4:5":
                width = int(base_height * 4 / 5)
                height = base_height
            elif aspect_ratio == "9:16":
                width = int(base_height * 9 / 16)
                height = base_height
        
        # Ensure width and height are even numbers for better compatibility
        width = width + (width % 2)
        height = height + (height % 2)
        
        return width, height
    
    def _transcribe_video(self, video_path: str, model_name: str = "base") -> List[Dict[str, Any]]:
        """Transcribe the video and return timestamped segments"""
        print(f"Transcribing video: {video_path}")
        
        # Extract audio from video
        audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            "-f", "wav",
            audio_path
        ]
        
        subprocess.run(cmd, check=True)
        
        if not os.path.exists(audio_path):
            raise RuntimeError("Failed to extract audio from video")
        
        # Load Whisper model if not already loaded
        if not self.whisper_model:
            print(f"Loading Whisper {model_name} model...")
            self.whisper_model = whisper.load_model(model_name)
        
        # Transcribe
        try:
            print("Starting transcription...")
            result = self.whisper_model.transcribe(audio_path)
            
            transcription = []
            for segment in result['segments']:
                transcription.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            
            # Clean up temporary audio file
            os.remove(audio_path)
            
            return transcription
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def _generate_captions_srt(self, transcription: List[Dict[str, Any]], output_path: str) -> str:
        """Generate SRT file from transcription"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(transcription):
                # Format time in SRT format (HH:MM:SS,mmm)
                start_time = self._format_srt_time(segment['start'])
                end_time = self._format_srt_time(segment['end'])
                
                # Write SRT entry
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
        
        return output_path
    
    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    
    def _create_caption_generator(self, transcription: List[Dict[str, Any]], style: Dict[str, Any]):
        """Creates a caption generator function for MoviePy"""
        def get_caption_at_time(t):
            # Find the segment that contains this timestamp
            for segment in transcription:
                if segment['start'] <= t <= segment['end']:
                    return segment['text']
            return ""
        
        return get_caption_at_time
    
    def _get_target_aspect_ratio_float(self, aspect_ratio: str) -> float:
        """Convert aspect ratio string to float value (width/height)"""
        width_ratio, height_ratio = ASPECT_RATIOS[aspect_ratio]
        return width_ratio / height_ratio
    
    def enhance_video(
        self,
        video_path: str,
        source_language: str,
        target_aspect_ratio: str,
        target_quality: str,
        provided_transcript: Optional[List[Dict[str, Any]]] = None,
        caption_style: Optional[Dict[str, Any]] = None,
        enable_debug: bool = False
    ) -> str:
        """
        Enhance a video by:
        1. Converting to target aspect ratio & resolution using intelligent reframing
        2. Adding burned-in captions
        
        Returns the path to the enhanced video
        """
        print(f"Enhancing video: {video_path}")
        print(f"Target aspect ratio: {target_aspect_ratio}")
        print(f"Target quality: {target_quality}")
        
        # Check if the file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Calculate the target dimensions
        target_width, target_height = self._get_target_dimensions(target_aspect_ratio, target_quality)
        print(f"Target dimensions: {target_width}x{target_height}")
        
        # Create output directory
        output_dir = "output_videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get base filename without extension
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_filename}_{target_aspect_ratio.replace(':', '_')}_{target_quality}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get transcript (use provided or generate new)
        transcript = provided_transcript
        if not transcript:
            transcript = self._transcribe_video(video_path)
            if not transcript:
                raise RuntimeError("Failed to generate transcript")
        
        # Generate SRT file for subtitle rendering
        srt_path = os.path.join(self.temp_dir, "captions.srt")
        self._generate_captions_srt(transcript, srt_path)
        
        # Apply intelligent reframing with subject tracking
        reframed_video_path = self._create_reframed_video(
            video_path, 
            target_aspect_ratio,
            target_width, 
            target_height,
            enable_debug
        )
        
        # Apply caption styling
        if caption_style is None:
            caption_style = DEFAULT_CAPTION_STYLE.copy()
        
        # Add burned-in captions to the reframed video
        caption_cmd = [
            self.ffmpeg_path,
            "-i", reframed_video_path,
            "-vf", f"subtitles={srt_path}:force_style='FontName={caption_style['font']},FontSize={caption_style['fontsize']},PrimaryColour=&H{self._rgb_to_hex(caption_style['color'])},BackColour=&H{self._rgb_to_hex(caption_style['bg_color'])}%{int(caption_style['bg_opacity']*255):X}'",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "copy",
            output_path
        ]
        
        print(f"Running caption command: {' '.join(caption_cmd)}")
        subprocess.run(caption_cmd, check=True)
        
        # Clean up temporary files
        if os.path.exists(reframed_video_path):
            os.remove(reframed_video_path)
        
        if os.path.exists(srt_path):
            os.remove(srt_path)
        
        print(f"Enhanced video created at: {output_path}")
        return output_path
    
    def _create_reframed_video(
        self,
        video_path: str,
        target_aspect_ratio: str,
        target_width: int,
        target_height: int,
        enable_debug: bool = False
    ) -> str:
        """
        Create a reframed video with intelligent subject tracking.
        Returns the path to the temporary reframed video file.
        """
        # Initialize subject tracker
        self.subject_tracker = SubjectTracker()
        
        # Get target aspect ratio as float (width/height)
        target_aspect_float = self._get_target_aspect_ratio_float(target_aspect_ratio)
        
        # Set up temporary file paths
        frame_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        
        reframed_video_path = os.path.join(self.temp_dir, "reframed.mp4")
        
        # Extract video information
        video_info_cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-hide_banner"
        ]
        
        try:
            video_info = subprocess.run(video_info_cmd, capture_output=True, text=True, check=False)
            
            # Extract frame rate
            fps_match = re.search(r'(\d+(?:\.\d+)?) fps', video_info.stderr)
            frame_rate = 30  # Default
            if fps_match:
                frame_rate = float(fps_match.group(1))
                print(f"Detected frame rate: {frame_rate} fps")
        except Exception as e:
            print(f"Warning: Could not determine frame rate: {e}")
            frame_rate = 30  # Fallback to 30 fps
        
        if enable_debug:
            # For debug mode, process the video frame by frame using OpenCV
            # This is slower but allows visualization of the tracking
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file with OpenCV")
            
            # Get video properties
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up debug output video
            debug_video_path = os.path.join(self.temp_dir, "debug_tracking.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(debug_video_path, fourcc, frame_rate, (orig_width, orig_height))
            
            # Set up crop params
            frame_idx = 0
            print(f"Processing video with debug visualization...")
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get crop window for this frame
                crop_box = self.subject_tracker.get_crop_window(frame, target_aspect_float)
                
                # Draw debug overlay
                debug_frame = self.subject_tracker.draw_debug_overlay(frame, crop_box)
                
                # Add frame number text
                cv2.putText(debug_frame, f"Frame: {frame_idx}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write to debug video
                debug_writer.write(debug_frame)
                
                # Apply crop to original frame
                x, y, w, h = crop_box
                cropped_frame = frame[y:y+h, x:x+w]
                resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
                
                # Save to frame directory
                cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg"), resized_frame)
                
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames...")
            
            cap.release()
            debug_writer.release()
            
            # Create video from frames using FFmpeg
            frames_to_video_cmd = [
                self.ffmpeg_path,
                "-y",
                "-framerate", str(frame_rate),
                "-i", os.path.join(frame_dir, "frame_%06d.jpg"),
                "-i", video_path,  # Original video for audio
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "copy",
                reframed_video_path
            ]
            
            print(f"Creating reframed video from frames...")
            subprocess.run(frames_to_video_cmd, check=True)
            
            print(f"Debug tracking video created at: {debug_video_path}")
            
        else:
            # For production mode, use FFmpeg's crop filter with scene detection
            # This is much faster but doesn't allow visualizing the tracking
            
            # Get aspect ratio of source video
            cap = cv2.VideoCapture(video_path)
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            source_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            
            # Perform initial frame analysis to set starting crop position
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError("Could not read first frame of video")
            
            # Initialize subject tracking with first frame
            self.subject_tracker.initialize_tracking(first_frame)
            x, y, w, h = self.subject_tracker.get_crop_window(first_frame, target_ratio)
            
            # Determine crop dimensions to maintain aspect ratio
            if source_ratio > target_ratio:  # Original is wider than target
                # Need to crop the sides
                crop_width = int(orig_height * target_ratio)
                crop_height = orig_height
                
                # Center crop on subject
                x_center = x + (w // 2)
                x_offset = max(0, min(x_center - (crop_width // 2), orig_width - crop_width))
                
                crop_filter = f"crop={crop_width}:{crop_height}:{x_offset}:0"
            else:  # Original is taller than or equal to target
                # Need to crop the top/bottom
                crop_width = orig_width
                crop_height = int(orig_width / target_ratio)
                
                # Center crop on subject
                y_center = y + (h // 2)
                y_offset = max(0, min(y_center - (crop_height // 2), orig_height - crop_height))
                
                crop_filter = f"crop={crop_width}:{crop_height}:0:{y_offset}"
            
            # Create the reframed video using FFmpeg
            reframe_cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vf", f"{crop_filter},scale={target_width}:{target_height}",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "128k",
                reframed_video_path
            ]
            
            print(f"Running reframe command: {' '.join(reframe_cmd)}")
            subprocess.run(reframe_cmd, check=True)
        
        return reframed_video_path
    
    def _rgb_to_hex(self, color_name: str) -> str:
        """Convert a color name to hex format for FFmpeg subtitles"""
        # Simple color mapping
        color_map = {
            "white": "FFFFFF",
            "black": "000000",
            "red": "FF0000",
            "green": "00FF00",
            "blue": "0000FF",
            "yellow": "FFFF00"
        }
        
        if color_name in color_map:
            return color_map[color_name]
        
        return "FFFFFF"  # Default to white
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

# Example usage function
def main():
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description="Enhance videos with aspect ratio conversion and captions")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--source-language", default="en-US", help="Source language for captioning")
    parser.add_argument("--aspect-ratio", choices=["1:1", "4:5", "9:16"], default="9:16", help="Target aspect ratio")
    parser.add_argument("--quality", choices=["1080p", "720p", "480p"], default="1080p", help="Target quality level")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    
    args = parser.parse_args()
    
    enhancer = VideoEnhancer()
    try:
        output_path = enhancer.enhance_video(
            args.video_path,
            args.source_language,
            args.aspect_ratio,
            args.quality,
            enable_debug=args.debug
        )
        print(f"Enhanced video saved to: {output_path}")
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    main() 