# CLIP: AI-Powered Video Enhancement System

CLIP is an advanced AI system for video enhancement and transformation, designed to help content creators optimize their videos for different social media platforms.

## Features

- **Smart Reframing**: Automatically reframe your videos to different aspect ratios (1:1, 4:5, 9:16) using AI to keep the important content in frame
- **Auto-Captioning**: Add professional burned-in captions to your videos with multiple style options
- **Clip Extraction**: Intelligently extract the most engaging clips from longer videos
- **Batch Processing**: Process multiple videos at once with the same settings
- **Multi-language Support**: Works with various source languages for captioning

## System Requirements

- Python 3.8 or higher
- FFmpeg installed and in PATH
- NVIDIA API key (for AI features)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/clip-video-enhancer.git
   cd clip-video-enhancer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your NVIDIA API key:
   ```
   export NVIDIA_API_KEY=your_api_key_here
   ```

## Quick Start

1. Place your videos in the `input_videos` directory
2. Run the enhancer:
   ```
   python video_enhancer_agent.py -i input_videos/your_video.mp4 -a 9:16 -q 720p
   ```
3. Find your enhanced videos in the `output_videos` directory

## Usage Guide

### Basic Command Structure

```
python video_enhancer_agent.py [options]
```

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-i, --input` | Input video path | `-i input_videos/video.mp4` |
| `-a, --aspect-ratio` | Target aspect ratio | `-a 9:16` |
| `-q, --quality` | Output quality | `-q 720p` |
| `-l, --language` | Source language | `-l en-US` |
| `-s, --caption-style` | Caption style | `-s bold` |
| `-b, --batch` | Batch process all videos | `-b` |
| `-e, --extract` | Extract clips | `-e` |

### Examples

#### Vertical (Stories/Reels) Format
```
python video_enhancer_agent.py -i input_videos/video.mp4 -a 9:16 -q 720p
```

#### Square (Instagram) Format
```
python video_enhancer_agent.py -i input_videos/video.mp4 -a 1:1 -q 1080p
```

#### Batch Processing
```
python video_enhancer_agent.py -b -a 4:5 -q 720p
```

## Directory Structure

- `input_videos/`: Place your source videos here
- `output_videos/`: Enhanced full videos are saved here
- `output_clips/`: Extracted clips are saved here

## Troubleshooting

### Common Issues

1. **Error: FFmpeg not found**
   - Make sure FFmpeg is installed and added to your PATH

2. **No captions appearing**
   - Check if the video has clear audio for transcription
   - Try specifying the language with the `-l` option

3. **Reframing issues**
   - For videos with fast-moving subjects, try using debug mode with `-d`

## Advanced Features

### Custom Caption Styles

You can customize the appearance of captions by editing the caption style parameters in `video_enhancer.py`.

### Clip Extraction

Use the interactive clip extraction mode to select specific portions of a video:

```
python video_enhancer_agent.py -e -i input_videos/your_video.mp4
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper transcription model
- NVIDIA for NeMo AI services
- MoviePy for video processing capabilities
