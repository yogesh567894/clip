import os
import sys
import subprocess
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import requests
import json
import ast
from openai import OpenAI

# Ensure output directory exists
def ensure_output_dir():
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    return output_dir

# Step 1: Transcribe the Video
def transcribe_video(video_path, model_name="base"):
    # First check for environment variable
    ffmpeg_env_path = os.environ.get('FFMPEG_PATH')
    ffmpeg_dir = None
    
    # Directly look for ffmpeg in the known essentials_build directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    essentials_build_path = os.path.join(current_dir, "ffmpeg-2025-04-21-git-9e1162bdf1-essentials_build", "bin", "ffmpeg.exe")
    
    if os.path.exists(essentials_build_path):
        print(f"Found FFmpeg in essentials_build directory: {essentials_build_path}")
        ffmpeg_env_path = essentials_build_path
        ffmpeg_dir = os.path.dirname(essentials_build_path)
        
        # Add to PATH for Whisper
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            print(f"Added FFmpeg directory to PATH: {ffmpeg_dir}")
    elif ffmpeg_env_path and os.path.exists(ffmpeg_env_path):
        # Extract the directory containing ffmpeg.exe
        ffmpeg_dir = os.path.dirname(ffmpeg_env_path)
        print(f"Adding FFmpeg directory to PATH: {ffmpeg_dir}")
        
        # Temporarily add ffmpeg directory to system PATH for Whisper
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    audio_path = "temp_audio.wav"
    
    # Process the video to wav for Whisper
    if ffmpeg_env_path and os.path.exists(ffmpeg_env_path):
        print(f"Using FFmpeg from environment variable: {ffmpeg_env_path}")
        
        # Fix path escaping for command line
        video_path_escaped = video_path.replace('"', '\\"')  # Escape any quotes in the path
        
        # Use different approach for Windows command line
        if os.name == 'nt':  # Windows
            ffmpeg_cmd = f'"{ffmpeg_env_path}" -i "{video_path}" -ar 16000 -ac 1 -b:a 64k -f wav temp_audio.wav'
        else:  # Unix/Linux/Mac
            ffmpeg_cmd = f'"{ffmpeg_env_path}" -i "{video_path_escaped}" -ar 16000 -ac 1 -b:a 64k -f wav "temp_audio.wav"'
            
        print(f"Running command: {ffmpeg_cmd}")
        result = os.system(ffmpeg_cmd)
        if result == 0 and os.path.exists(audio_path):
            # FFmpeg worked, so proceed with transcription
            pass
        else:
            print(f"Failed to use FFmpeg from environment variable with error code: {result}. Will try other methods.")
            
            # Try with subprocess instead of os.system as a fallback
            print("Trying with subprocess instead...")
            try:
                subprocess_cmd = [ffmpeg_env_path, "-i", video_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-f", "wav", audio_path]
                print(f"Running subprocess command: {' '.join(subprocess_cmd)}")
                subprocess.run(subprocess_cmd, check=True)
                if os.path.exists(audio_path):
                    print("Subprocess approach worked!")
            except Exception as e:
                print(f"Subprocess approach failed with error: {e}")
    
    # If we don't have a successful run from env variable, try finding it locally
    if not os.path.exists(audio_path):
        # Set the path to ffmpeg in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for ffmpeg with different possible filenames
        possible_ffmpeg_names = ["ffmpeg.exe", "ffmpeg-win64.exe", "ffmpeg-win32.exe", "ffmpeg"]
        ffmpeg_found = False
        ffmpeg_path = None
        
        print(f"Looking for FFmpeg in: {current_dir}")
        for name in possible_ffmpeg_names:
            test_path = os.path.join(current_dir, name)
            if os.path.exists(test_path):
                # Check if it's a file, not a directory
                if os.path.isfile(test_path):
                    ffmpeg_path = test_path
                    ffmpeg_found = True
                    print(f"Found FFmpeg at: {ffmpeg_path}")
                    break
                else:
                    print(f"Found '{name}' but it's a directory, not an executable")
                    # Check if there's ffmpeg.exe inside this directory
                    possible_exe = os.path.join(test_path, "ffmpeg.exe")
                    if os.path.exists(possible_exe) and os.path.isfile(possible_exe):
                        ffmpeg_path = possible_exe
                        ffmpeg_found = True
                        print(f"Found FFmpeg at: {ffmpeg_path}")
                        break
                    
                    # Also check in bin subdirectory which is common
                    bin_exe = os.path.join(test_path, "bin", "ffmpeg.exe")
                    if os.path.exists(bin_exe) and os.path.isfile(bin_exe):
                        ffmpeg_path = bin_exe
                        ffmpeg_found = True
                        print(f"Found FFmpeg at: {ffmpeg_path}")
                        
                        # Add this bin directory to PATH for Whisper
                        bin_dir = os.path.dirname(bin_exe)
                        if bin_dir not in os.environ["PATH"]:
                            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                            print(f"Added {bin_dir} to PATH for Whisper")
                        
                        break
        
        if not ffmpeg_found:
            # List files in directory to help debugging
            print("FFmpeg not found. Files in current directory:")
            for file in os.listdir(current_dir):
                if "ffmpeg" in file.lower():
                    print(f" - {file}  <-- POTENTIAL MATCH")
                else:
                    print(f" - {file}")
            
            # Try using ffmpeg from PATH as last resort
            try:
                print("Trying system PATH for FFmpeg...")
                subprocess_cmd = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-f", "wav", audio_path]
                print(f"Running subprocess command: {' '.join(subprocess_cmd)}")
                subprocess.run(subprocess_cmd, check=True)
                if os.path.exists(audio_path):
                    print("Using FFmpeg from system PATH worked!")
                    result = 0
                else:
                    result = 1
            except Exception as e:
                print(f"System PATH approach failed with error: {e}")
                result = 1
        else:
            # Use the found ffmpeg executable with subprocess
            try:
                print(f"Using found FFmpeg: {ffmpeg_path}")
                subprocess_cmd = [ffmpeg_path, "-i", video_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-f", "wav", audio_path]
                print(f"Running subprocess command: {' '.join(subprocess_cmd)}")
                subprocess.run(subprocess_cmd, check=True)
                if os.path.exists(audio_path):
                    print("Using local FFmpeg worked!")
                    
                    # Add the found FFmpeg directory to PATH for Whisper
                    ffmpeg_dir = os.path.dirname(ffmpeg_path)
                    if ffmpeg_dir not in os.environ["PATH"]:
                        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
                        print(f"Added {ffmpeg_dir} to PATH for Whisper")
                    
                    result = 0
                else:
                    result = 1
            except Exception as e:
                print(f"Local FFmpeg approach failed with error: {e}")
                result = 1
    
    if not os.path.exists(audio_path):
        print(f"Error: Failed to create audio file. FFmpeg command failed with code: {result}")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure FFmpeg is downloaded and in the same folder as this script")
        print("2. Rename the FFmpeg executable to 'ffmpeg.exe'")
        print("3. Or add FFmpeg to your system PATH")
        print("4. Or set the FFMPEG_PATH environment variable to the full path to ffmpeg.exe")
        print("\nExample to set environment variable:")
        print("   In PowerShell: $env:FFMPEG_PATH = 'C:\\path\\to\\ffmpeg.exe'")
        print("   In CMD: set FFMPEG_PATH=C:\\path\\to\\ffmpeg.exe")
        return None
    
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    
    print("Starting transcription...")
    print(f"FFmpeg should be found in PATH: {os.environ['PATH']}")
    print(f"Audio file exists: {os.path.exists(audio_path)}")
    
    try:
        result = model.transcribe(audio_path)
        transcription = []
        for segment in result['segments']:
            transcription.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        # Clean up temporary audio file
        try:
            os.remove(audio_path)
        except:
            pass
            
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Trying with direct audio path...")
        
        # Try a different approach by directly loading the audio file
        try:
            import numpy as np
            from whisper.audio import load_audio
            audio = np.zeros((1,))  # Initialize with empty array
            
            # Load audio directly without using FFmpeg
            import soundfile as sf
            if os.path.exists(audio_path):
                audio, _ = sf.read(audio_path)
                # Convert to float32 and normalize
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32) / (1 << 15)
                
                # Convert to mono if needed
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 16kHz if needed
                if _ != 16000:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * 16000 / _))
                
                result = model.transcribe(audio)
                transcription = []
                for segment in result['segments']:
                    transcription.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip()
                    })
                return transcription
            else:
                print(f"Audio file not found at: {audio_path}")
                return None
        except Exception as e2:
            print(f"Error during alternative transcription: {e2}")
            return None

def get_relevant_segments(transcript, user_query):
    # Check if this is a request for a short clip with time limit
    is_short_clip_request = "short clip" in user_query.lower() or "short video" in user_query.lower()
    time_limit = 60  # Default to 60 seconds if not specified
    
    # Try to find a time limit in the query
    import re
    time_limit_match = re.search(r'(\d+)\s*seconds', user_query.lower())
    if time_limit_match:
        time_limit = int(time_limit_match.group(1))
        print(f"Detected time limit request: {time_limit} seconds")
    
    # Customize the prompt based on the request type
    if is_short_clip_request:
        prompt = f"""You are an expert video editor who can read video transcripts and create short clips. Given a transcript with segments, your task is to identify a single compelling segment or conversation that would make an excellent short clip.

Guidelines:
1. The clip should be relevant to the user query and highly engaging.
2. The clip should be coherent and self-contained, with a beginning, middle, and end.
3. The clip must be {time_limit} seconds or less in total duration.
4. Choose the most interesting, visually compelling, or emotionally resonant segment.
5. Do not cut off in the middle of a sentence or idea.
6. Match the start and end time of the segment using the timestamps from the transcript.

Output format (IMPORTANT): You must respond ONLY with valid JSON in this format:
{{"conversations": [{{"start": start_time_in_seconds, "end": end_time_in_seconds}}]}}

Transcript:
{transcript}

User query:
{user_query}"""
    else:
        prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. Follow these guidelines when choosing conversations. A group of continuous segments in the transcript is a conversation.

Guidelines:
1. The conversation should be relevant to the user query. The conversation should include more than one segment to provide context and continuity.
2. Include all the before and after segments needed in a conversation to make it complete.
3. The conversation should not cut off in the middle of a sentence or idea.
4. Choose multiple conversations from the transcript that are relevant to the user query.
5. Match the start and end time of the conversations using the segment timestamps from the transcript.
6. The conversations should be a direct part of the video and should not be out of context.

Output format (IMPORTANT): You must respond ONLY with valid JSON in this format:
{{"conversations": [{{"start": start_time_in_seconds, "end": end_time_in_seconds}}, {{"start": start_time_in_seconds, "end": end_time_in_seconds}}]}}

Transcript:
{transcript}

User query:
{user_query}"""

    # Use NVIDIA API with Llama 3 model
    try:
        print("Using NVIDIA API with Llama 3...")
        
        # Set up OpenAI client with NVIDIA API
        nvidia_api_key = os.environ.get('NVIDIA_API_KEY', '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')
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        # Create the chat completion
        print("Sending request to NVIDIA API...")
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert video editor assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=1,
            max_tokens=1024
        )
        
        # Get the response content
        content = completion.choices[0].message.content
        print(f"API Response Content: {content[:200]}...")  # Print first 200 chars
        
        # Extract just the JSON part if it's embedded in text
        import re
        json_match = re.search(r'(\{.*"conversations":\s*\[.*\].*\})', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"Extracted JSON: {json_str[:200]}...")
        else:
            # Try to find JSON within code blocks
            json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print(f"Extracted JSON from code block: {json_str[:200]}...")
            else:
                json_str = content
        
        # Try multiple approaches to parse the JSON
        conversations = None
        
        try:
            # Try the most straightforward approach first
            if '{"conversations":' in json_str:
                try:
                    # Try with json.loads first (preferred)
                    cleaned_json = json_str.replace("```json", "").replace("```", "").strip()
                    conversations = json.loads(cleaned_json)["conversations"]
                except (json.JSONDecodeError, KeyError):
                    # Fall back to ast.literal_eval
                    conversations = ast.literal_eval(json_str)["conversations"]
            
            # If that didn't work, try regex approaches
            if not conversations:
                # Try to extract timestamps with regex
                pattern = r'"start"\s*:\s*"?(\d+(?:\.\d+)?)"?\s*,\s*"end"\s*:\s*"?(\d+(?:\.\d+)?)"?'
                matches = re.findall(pattern, json_str)
                
                if matches:
                    print(f"Found {len(matches)} pairs of timestamps using regex")
                    conversations = [{"start": float(start), "end": float(end)} for start, end in matches]
                else:
                    # Another fallback regex for different format
                    pattern2 = r'"start"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"end"\s*:\s*(\d+(?:\.\d+)?)'
                    matches2 = re.findall(pattern2, json_str)
                    
                    if matches2:
                        print(f"Found {len(matches2)} pairs of timestamps using second regex pattern")
                        conversations = [{"start": float(start), "end": float(end)} for start, end in matches2]
            
            # If we found segments, validate them
            if conversations:
                validated_segments = []
                for seg in conversations:
                    # Convert string values to float if needed
                    start = float(seg["start"]) if isinstance(seg["start"], str) else seg["start"]
                    end = float(seg["end"]) if isinstance(seg["end"], str) else seg["end"]
                    
                    # Ensure start and end are within the video bounds
                    start = max(start, transcript[0]["start"])
                    end = min(end, transcript[-1]["end"])
                    
                    # Skip invalid segments
                    if end <= start:
                        continue
                    
                    # For short clip requests, enforce the time limit
                    if is_short_clip_request and end - start > time_limit:
                        print(f"Segment too long ({end - start:.2f}s), trimming to {time_limit}s")
                        # Keep the first part of the segment up to the time limit
                        end = start + time_limit
                    
                    validated_segments.append({"start": start, "end": end})
                
                if validated_segments:
                    print(f"Successfully parsed and validated {len(validated_segments)} segments")
                    return validated_segments
        
        except (SyntaxError, ValueError, KeyError) as e:
            print(f"Error parsing content as JSON: {e}")
        
        # Last resort fallback
        print("No valid segments found. Creating fallback segment...")
        if transcript:
            # Check if the user requested a short clip with a time limit
            if "short clip" in user_query.lower() or "short video" in user_query.lower():
                # Try to find a time limit in the query
                import re
                time_limit_match = re.search(r'(\d+)\s*seconds', user_query.lower())
                time_limit = 60  # Default to 60 seconds if not specified
                
                if time_limit_match:
                    time_limit = int(time_limit_match.group(1))
                    print(f"Detected time limit request: {time_limit} seconds")
                else:
                    print(f"Using default short clip duration: {time_limit} seconds")
                
                # Find an interesting segment in the middle of the video
                total_duration = transcript[-1]["end"] - transcript[0]["start"]
                middle_point = transcript[0]["start"] + (total_duration / 2)
                
                # Create a segment centered around the middle point with the requested duration
                half_duration = min(time_limit / 2, total_duration / 2)
                start_time = max(transcript[0]["start"], middle_point - half_duration)
                end_time = min(transcript[-1]["end"], start_time + time_limit)
                
                print(f"Creating a {end_time - start_time:.2f} second clip from {start_time:.2f} to {end_time:.2f}")
                return [{"start": start_time, "end": end_time}]
            
            # If no short clip requested, use the entire video
            return [{"start": transcript[0]["start"], "end": transcript[-1]["end"]}]
        return []
    
    except Exception as e:
        print(f"Error with NVIDIA API: {e}")
        # Fallback option: create segment for entire video or a short clip
        if transcript:
            # Check if the user requested a short clip with a time limit
            if "short clip" in user_query.lower() or "short video" in user_query.lower():
                # Try to find a time limit in the query
                import re
                time_limit_match = re.search(r'(\d+)\s*seconds', user_query.lower())
                time_limit = 60  # Default to 60 seconds if not specified
                
                if time_limit_match:
                    time_limit = int(time_limit_match.group(1))
                    print(f"Detected time limit request: {time_limit} seconds")
                else:
                    print(f"Using default short clip duration: {time_limit} seconds")
                
                # Find an interesting segment in the middle of the video
                total_duration = transcript[-1]["end"] - transcript[0]["start"]
                middle_point = transcript[0]["start"] + (total_duration / 2)
                
                # Create a segment centered around the middle point with the requested duration
                half_duration = min(time_limit / 2, total_duration / 2)
                start_time = max(transcript[0]["start"], middle_point - half_duration)
                end_time = min(transcript[-1]["end"], start_time + time_limit)
                
                print(f"Creating a {end_time - start_time:.2f} second clip from {start_time:.2f} to {end_time:.2f}")
                return [{"start": start_time, "end": end_time}]
            
            # If no short clip requested, use the entire video
            return [{"start": transcript[0]["start"], "end": transcript[-1]["end"]}]
        return []

def edit_video(original_video_path, segments, output_video_path, fade_duration=0.5):
    video = VideoFileClip(original_video_path)
    clips = []
    for seg in segments:
        start = seg['start']
        end = seg['end']
        clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)
        clips.append(clip)
    if clips:
        try:
            # Fix output path to avoid special characters
            import re
            
            # Get the directory and basename
            output_dir = os.path.dirname(output_video_path)
            basename = os.path.basename(output_video_path)
            
            # Clean the filename - remove special characters
            clean_basename = re.sub(r'[^\w\-\.]', '_', basename)
            
            # If the filename changed, update the path
            if clean_basename != basename:
                print(f"Sanitizing output filename from '{basename}' to '{clean_basename}'")
                output_video_path = os.path.join(output_dir, clean_basename)
            
            print(f"Creating final video at: {output_video_path}")
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # First try with normal settings
            try:
                final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
            except OSError as e:
                print(f"First attempt failed: {e}")
                
                # Try a more basic approach
                print("Trying alternate method...")
                temp_path = os.path.join(output_dir, "temp_output.mp4")
                final_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", 
                                         ffmpeg_params=["-pix_fmt", "yuv420p", "-strict", "-2"])
                
                # If successful, try to rename
                if os.path.exists(temp_path):
                    try:
                        if os.path.exists(output_video_path):
                            os.remove(output_video_path)
                        os.rename(temp_path, output_video_path)
                        print(f"Successfully created video at: {output_video_path}")
                    except OSError:
                        print(f"Could not rename file, but video was created at: {temp_path}")
            
            final_clip.close()
        except Exception as e:
            print(f"Error editing video: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No segments to include in the edited video.")
    
    # Close the original video file
    video.close()

# Main Function
def main():
    # Setup input and output folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_videos")
    output_dir = ensure_output_dir()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please create this directory and place your videos inside it.")
        return
    
    # List available videos in the input directory
    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"No videos found in '{input_dir}' directory.")
        print("Please add video files (mp4, avi, mov, mkv) to this directory.")
        return
    
    print("Available videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {video}")
    
    # Let user choose a video or use the first one
    choice = 1  # Default to first video
    try:
        choice = int(input(f"Select a video (1-{len(videos)}) or press Enter for default: ") or "1")
        if choice < 1 or choice > len(videos):
            print(f"Invalid choice. Using first video.")
            choice = 1
    except ValueError:
        print(f"Invalid input. Using first video.")
        choice = 1
    
    selected_video = videos[choice-1]
    
    # Use absolute paths to prevent resolution issues
    input_video = os.path.abspath(os.path.join(input_dir, selected_video))
    
    # Generate a clean filename for output
    import re
    clean_filename = re.sub(r'[^\w\-\.]', '_', f"edited_{selected_video}")
    output_video = os.path.abspath(os.path.join(output_dir, clean_filename))
    
    print(f"Selected: {selected_video}")
    
    # User Query
    default_query = "Find all clips where there is discussion around GPT-4 Turbo"
    user_query = input(f"Enter your search query or press Enter for default: ") or default_query
    
    print(f"Using query: {user_query}")
    print(f"Input video: {input_video}")
    print(f"Output will be saved to: {output_video}")
    
    # Step 1: Transcribe
    print("Transcribing video...")
    transcription = transcribe_video(input_video, model_name="base")
    
    if transcription is None:
        print("Transcription failed. Please check the FFmpeg installation and try again.")
        return
    
    print("Transcription completed successfully.")
    print("Finding relevant segments...")
    relevant_segments = get_relevant_segments(transcription, user_query)
    
    if not relevant_segments:
        print("No relevant segments found matching your query.")
        return
    
    print(f"Found {len(relevant_segments)} relevant segments.")
    
    # Step 5: Edit Video
    print("Editing video...")
    edit_video(input_video, relevant_segments, output_video)
    print(f"Edited video processing complete.")

if __name__ == "__main__":
    main()
