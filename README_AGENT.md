# ClipAnything Agent

This is an enhanced version of ClipAnything that uses LangChain and AI agents to create compelling story-like clips from your videos.

## Features

- **AI-powered clip selection**: Uses LangChain and Llama 3 to intelligently select the most relevant segments
- **Story-driven editing**: Creates clips with narrative structure - beginning, middle, and end
- **Customizable number of clips**: Choose how many clips you want in your final video
- **Automatic transcription**: Uses Whisper to transcribe your videos for content analysis
- **Title overlays**: Adds professional title overlays to each clip segment
- **Story introduction**: Creates a proper introduction and conclusion for your video story

## Requirements

- Python 3.8 or higher
- FFmpeg installed (included in the repository)
- NVIDIA API key (for LLM access)
- ImageMagick (optional, for text overlays)

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set the FFMPEG_PATH environment variable (optional):
   ```
   # Windows PowerShell
   $env:FFMPEG_PATH="C:\path\to\ffmpeg.exe"
   
   # Windows CMD
   set FFMPEG_PATH=C:\path\to\ffmpeg.exe
   ```

3. Set your NVIDIA API key (if using a different key):
   ```
   # Windows PowerShell
   $env:NVIDIA_API_KEY="your-api-key-here"
   
   # Windows CMD
   set NVIDIA_API_KEY=your-api-key-here
   ```

4. Install ImageMagick (optional, for text overlays):
   - Download from: https://imagemagick.org/script/download.php
   - During installation, make sure to check "Install legacy utilities (convert)"
   - The app will try to auto-detect ImageMagick or work without it

## Usage

1. Run the agent script:
   ```
   python agent_app.py
   ```

2. Select a video from your input_videos folder
3. Enter your search query (what kind of content you want to extract)
4. Specify how many clips you want to include
5. Wait for the AI to analyze the video and create your story

## How It Works

1. **Video Transcription**: The system first transcribes your video using OpenAI's Whisper model
2. **AI Analysis**: The transcript is analyzed by a LangChain agent powered by Llama 3 70B
3. **Clip Selection**: The agent identifies the most relevant and engaging segments
4. **Story Structuring**: The clips are arranged in a narrative structure
5. **Video Editing**: The final video is created with titles, transitions, and a story-like format

## Example Queries

- "Create a short story about the most interesting parts of this video"
- "Extract all the funny moments and make them into a comedy sequence"
- "Create a tutorial from the key educational points in this lecture"
- "Make a dramatic story with the most emotional moments" 