#!/usr/bin/env python
import os
import sys
import argparse
from video_enhancer import VideoEnhancer

def main():
    parser = argparse.ArgumentParser(description="AI Video Enhancement Processor")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--source-language", default="en-US", help="Source language code for captioning (e.g., en-US, es-ES)")
    parser.add_argument("--aspect-ratio", choices=["1:1", "4:5", "9:16"], default="9:16", 
                        help="Target aspect ratio for the processed video")
    parser.add_argument("--quality", choices=["1080p", "720p", "480p"], default="1080p", 
                        help="Target quality level for the processed video")
    parser.add_argument("--transcript", help="Path to optional transcript file (JSON format)")
    parser.add_argument("--caption-font", default="Arial", help="Font to use for captions")
    parser.add_argument("--caption-size", type=int, default=30, help="Font size for captions")
    parser.add_argument("--caption-color", default="white", help="Text color for captions")
    parser.add_argument("--caption-bg-color", default="black", help="Background color for captions")
    parser.add_argument("--caption-bg-opacity", type=float, default=0.5, help="Background opacity for captions (0-1)")
    parser.add_argument("--caption-position", default="bottom", choices=["top", "middle", "bottom"], 
                        help="Vertical position of captions")
    
    args = parser.parse_args()
    
    # Validate input file
    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Load optional transcript if provided
    transcript = None
    if args.transcript and os.path.exists(args.transcript):
        import json
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
    
    print(f"Enhancing video with the following settings:")
    print(f"  - Source language: {args.source_language}")
    print(f"  - Target aspect ratio: {args.aspect_ratio}")
    print(f"  - Target quality: {args.quality}")
    print(f"  - Caption style: {caption_style}")
    
    # Process the video
    enhancer = VideoEnhancer()
    try:
        output_path = enhancer.enhance_video(
            video_path,
            args.source_language,
            args.aspect_ratio,
            args.quality,
            provided_transcript=transcript,
            caption_style=caption_style
        )
        print(f"\nEnhancement completed!")
        print(f"Enhanced video saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error enhancing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 