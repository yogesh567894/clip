# AI Video Enhancement Processor

This tool provides advanced video processing capabilities to transform standard videos into different aspect ratios with burned-in captions. The system uses AI-powered subject detection and tracking to intelligently reframe content, ensuring the most important elements remain in frame.

## Features

- **Intelligent Reframing**: Automatically detects and tracks subjects to maintain optimal composition when converting between aspect ratios
- **Accurate Captioning**: Uses OpenAI's Whisper for high-quality speech transcription
- **Custom Caption Styling**: Configure font, size, color, and position of burned-in captions
- **Multiple Target Formats**: Support for popular social media aspect ratios (1:1, 4:5, 9:16)
- **Quality Control**: Choose output resolution based on target platform requirements

## Requirements

- Python 3.8+
- FFmpeg
- OpenCV
- Whisper model from OpenAI

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/video-enhancer.git
cd video-enhancer
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Ensure FFmpeg is installed on your system or included in the project directory.

## Usage

### Command Line

Basic usage:

```bash
python enhance_video.py path/to/video.mp4
```

With options:

```bash
python enhance_video.py path/to/video.mp4 --aspect-ratio 9:16 --quality 1080p --source-language en-US
```

### Arguments

- `video_path`: Path to the input video file
- `--source-language`: Language code for captioning (default: en-US)
- `--aspect-ratio`: Target aspect ratio (choices: 1:1, 4:5, 9:16, default: 9:16)
- `--quality`: Target quality level (choices: 1080p, 720p, 480p, default: 1080p)
- `--debug`: Enable debug visualization

### Caption Customization

```bash
python enhance_video.py path/to/video.mp4 --caption-font Arial --caption-size 30 --caption-color white --caption-bg-color black --caption-bg-opacity 0.5 --caption-position bottom
```

## Resolution Reference

The system will calculate the exact dimensions based on the selected aspect ratio and quality level:

**1:1 (Square)**
- 1080p: 1080x1080
- 720p: 720x720
- 480p: 480x480

**4:5 (Instagram Portrait)**
- 1080p: 864x1080
- 720p: 576x720
- 480p: 384x480

**9:16 (Portrait/Stories/Reels)**
- 1080p: 608x1080
- 720p: 405x720
- 480p: 270x480

## Advanced Usage

### Using a Pre-Existing Transcript

If you already have a transcript in JSON format, you can provide it to skip the transcription step:

```bash
python enhance_video.py path/to/video.mp4 --transcript path/to/transcript.json
```

The transcript should be in the following format:

```json
[
  {
    "start": 0.0,
    "end": 2.5,
    "text": "Hello and welcome to this video"
  },
  {
    "start": 2.6,
    "end": 5.2,
    "text": "Today we'll be discussing..."
  }
]
```

### Debug Mode

To visualize the subject tracking and cropping process:

```bash
python enhance_video.py path/to/video.mp4 --debug
```

This will generate a debug video showing the detected subjects and crop windows.

## Programmatic API

You can also use the VideoEnhancer class in your own Python code:

```python
from video_enhancer import VideoEnhancer

enhancer = VideoEnhancer()
try:
    output_path = enhancer.enhance_video(
        video_path="input.mp4",
        source_language="en-US",
        target_aspect_ratio="9:16",
        target_quality="1080p",
        caption_style={
            "font": "Arial",
            "fontsize": 40,
            "color": "white",
            "bg_color": "black",
            "bg_opacity": 0.7,
            "position": "bottom"
        }
    )
    print(f"Enhanced video saved to: {output_path}")
finally:
    enhancer.cleanup()
```

## How It Works

1. **Transcription**: Uses Whisper AI to transcribe speech to text with timestamps
2. **Subject Detection**: Identifies faces and other important subjects in the video
3. **Subject Tracking**: Follows detected subjects across frames to maintain consistent framing
4. **Intelligent Cropping**: Determines optimal crop window based on subject position and target aspect ratio
5. **Caption Rendering**: Converts transcription to SRT format and burns captions into the video

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Using CLIP: The Video Enhancement System

CLIP is an AI-powered video enhancement system that can transform your videos by:
1. Auto-reframing for different aspect ratios (1:1, 4:5, 9:16)
2. Adding burned-in captions
3. Extracting and enhancing clips from longer videos
4. Batch processing multiple videos

### Getting Started

Make sure you have all the dependencies installed:

```
pip install -r requirements.txt
```

### Basic Usage

The simplest way to enhance a video is to use the `video_enhancer_agent.py` script:

```
python video_enhancer_agent.py -i input_videos/your_video.mp4 -a 9:16 -q 720p -l en-US
```

This will:
- Take your input video
- Reframe it to 9:16 aspect ratio (vertical video)
- Set quality to 720p
- Add captions using English (US) as the source language
- Save the output to the `output_videos` directory

### Input and Output Directories

- Place input videos in the `input_videos` folder
- Enhanced videos are saved to `output_videos` folder
- Extracted clips are saved to `output_clips` folder

### Command Line Options

```
python video_enhancer_agent.py [options]

Options:
  -i, --input            Input video path or directory
  -a, --aspect-ratio     Target aspect ratio (1:1, 4:5, 9:16)
  -q, --quality          Target quality (1080p, 720p, 480p)
  -l, --language         Source language (en-US, es, fr, etc.)
  -s, --caption-style    Caption style (default, minimal, bold)
  -b, --batch            Process all videos in input directory
  -e, --extract          Extract clips from video (interactive mode)
  -d, --debug            Enable debug visualization
```

### Examples

1. **Process a video with square aspect ratio:**
   ```
   python video_enhancer_agent.py -i input_videos/your_video.mp4 -a 1:1 -q 720p
   ```

2. **Batch process all videos in a directory:**
   ```
   python video_enhancer_agent.py -b -a 9:16 -q 720p
   ```

3. **Extract clips from a video (interactive mode):**
   ```
   python video_enhancer_agent.py -e -i input_videos/your_video.mp4
   ```

4. **Use a specific caption style:**
   ```
   python video_enhancer_agent.py -i input_videos/your_video.mp4 -a 4:5 -s bold
   ```

### Interactive Mode

For a guided experience, simply run:

```
python video_enhancer_agent.py
```

This will start the interactive mode that will guide you through:
- Selecting input videos
- Choosing aspect ratio and quality
- Extracting clips or batch processing

### Troubleshooting

If you encounter issues:

1. Ensure FFmpeg is installed and in your PATH
2. Check that all dependencies are installed
3. Verify input videos are in a compatible format (MP4, AVI, MOV, MKV)
4. Make sure you have sufficient disk space for output videos 