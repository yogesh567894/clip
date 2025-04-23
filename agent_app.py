import os
import sys
import subprocess
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import json
from langchain_core.tools import BaseTool, Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import List, Dict, Any, Optional
import re

# NVIDIA API key
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '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')

# Initialize the LLM
llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct", 
    api_key=NVIDIA_API_KEY,
    streaming=False  # Disable streaming to avoid the chunk error
)

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
    
    if not ffmpeg_env_path or not os.path.exists(ffmpeg_env_path):
        # Set the path to ffmpeg in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_path = os.path.join(current_dir, "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
        
        if os.path.exists(ffmpeg_path):
            ffmpeg_env_path = ffmpeg_path
            os.environ['FFMPEG_PATH'] = ffmpeg_path
            print(f"Found FFmpeg at: {ffmpeg_path}")
        else:
            print("FFmpeg not found. Please set FFMPEG_PATH environment variable.")
            return None
    
    # Add ffmpeg directory to PATH for Whisper
    ffmpeg_dir = os.path.dirname(ffmpeg_env_path)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    audio_path = "temp_audio.wav"
    
    # Process the video to wav for Whisper
    try:
        print(f"Extracting audio using FFmpeg from: {video_path}")
        subprocess_cmd = [ffmpeg_env_path, "-i", video_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-f", "wav", audio_path]
        subprocess.run(subprocess_cmd, check=True)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None
    
    if not os.path.exists(audio_path):
        print("Failed to create audio file")
        return None
        
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    
    print("Starting transcription...")
    
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
        return None

# Tool for analyzing transcript and generating clip segments
class TranscriptAnalysisTool(BaseTool):
    name = "analyze_transcript"
    description = "Analyzes video transcript and returns segments to include in a video"
    
    def _run(self, transcript: List[Dict], user_query: str, num_clips: int = 1) -> Dict[str, Any]:
        print(f"Analyzing transcript for {num_clips} clips based on query: '{user_query}'")
        
        # Create prompt for transcript analysis
        prompt = f"""You are an expert video editor who can read video transcripts and create engaging story-like clips. Given a transcript with segments, identify {num_clips} compelling segments that would make excellent clips related to the user's query.

Guidelines:
1. Identify exactly {num_clips} segments that are relevant to the user query.
2. Each clip should be coherent and self-contained with a beginning, middle, and end.
3. The clips should tell a short story when played in sequence.
4. Each clip should be 30-60 seconds in duration.
5. Do not cut off in the middle of a sentence or idea.
6. Choose segments that are visually interesting or emotionally engaging.
7. If possible, create a narrative arc across all clips.

Output format (IMPORTANT): You must respond ONLY with valid JSON in this format:
{{
  "clips": [
    {{
      "start": start_time_in_seconds,
      "end": end_time_in_seconds,
      "title": "Brief title for this segment",
      "description": "Short description of why this segment is important"
    }},
    ...additional clips...
  ],
  "narrative_structure": "Brief explanation of how these clips form a story"
}}

Transcript:
{json.dumps(transcript, indent=2)}

User query:
{user_query}"""

        try:
            # Call the LLM for analysis
            response = llm.invoke(prompt)
            
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                # Try to convert to string
                content = str(response)
            
            print(f"LLM Response: {content[:200]}...")  # Print first 200 chars
            
            # Extract JSON from response
            json_match = re.search(r'(\{.*"clips":\s*\[.*\].*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON within code blocks
                json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = content
            
            # Parse JSON
            try:
                # Clean up the string for JSON parsing
                cleaned_json = json_str.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned_json)
                
                # Validate the clips
                validated_clips = []
                for clip in result.get("clips", []):
                    # Ensure required fields
                    if "start" not in clip or "end" not in clip:
                        continue
                    
                    # Convert to float if needed
                    start = float(clip["start"]) if isinstance(clip["start"], str) else clip["start"]
                    end = float(clip["end"]) if isinstance(clip["end"], str) else clip["end"]
                    
                    # Ensure start and end are valid
                    if end <= start:
                        continue
                    
                    # Add default title/description if missing
                    if "title" not in clip:
                        clip["title"] = f"Clip from {start:.1f}s to {end:.1f}s"
                    if "description" not in clip:
                        clip["description"] = "Relevant segment for the query"
                    
                    validated_clips.append({
                        "start": start,
                        "end": end,
                        "title": clip["title"],
                        "description": clip["description"]
                    })
                
                if not validated_clips and transcript:
                    # Fallback: create generic clips
                    print("No valid clips found. Creating fallback clips...")
                    total_duration = transcription[-1]["end"] - transcription[0]["start"]
                    clip_duration = min(60, total_duration / num_clips)
                    
                    for i in range(num_clips):
                        start_time = transcription[0]["start"] + (i * clip_duration)
                        end_time = min(transcription[-1]["end"], start_time + clip_duration)
                        
                        validated_clips.append({
                            "start": start_time,
                            "end": end_time,
                            "title": f"Clip {i+1}",
                            "description": f"Auto-generated clip {i+1}"
                        })
                
                # Add narrative structure
                narrative = result.get("narrative_structure", "A sequence of clips related to the query")
                
                return {
                    "clips": validated_clips,
                    "narrative_structure": narrative
                }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                # Fallback using regex
                clips = []
                matches = re.findall(r'"start"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"end"\s*:\s*(\d+(?:\.\d+)?)', json_str)
                
                if matches and transcription:
                    for i, (start, end) in enumerate(matches[:num_clips]):
                        clips.append({
                            "start": float(start),
                            "end": float(end),
                            "title": f"Clip {i+1}",
                            "description": f"Extracted clip {i+1}"
                        })
                
                if not clips and transcription:
                    # Fallback: create generic clips
                    total_duration = transcription[-1]["end"] - transcription[0]["start"]
                    clip_duration = min(60, total_duration / num_clips)
                    
                    for i in range(num_clips):
                        start_time = transcription[0]["start"] + (i * clip_duration)
                        end_time = min(transcription[-1]["end"], start_time + clip_duration)
                        
                        clips.append({
                            "start": start_time,
                            "end": end_time,
                            "title": f"Clip {i+1}",
                            "description": f"Auto-generated clip {i+1}"
                        })
                
                return {
                    "clips": clips,
                    "narrative_structure": "Automatically generated clip sequence"
                }
                
        except Exception as e:
            print(f"Error in transcript analysis: {e}")
            if transcription:
                # Create fallback clips
                clips = []
                total_duration = transcription[-1]["end"] - transcription[0]["start"]
                clip_duration = min(60, total_duration / num_clips)
                
                for i in range(num_clips):
                    start_time = transcription[0]["start"] + (i * clip_duration)
                    end_time = min(transcription[-1]["end"], start_time + clip_duration)
                    
                    clips.append({
                        "start": start_time,
                        "end": end_time,
                        "title": f"Clip {i+1}",
                        "description": f"Auto-generated clip {i+1}"
                    })
                
                return {
                    "clips": clips,
                    "narrative_structure": "Automatically generated clip sequence"
                }
            return {"clips": [], "narrative_structure": "Failed to analyze transcript"}

# Check if ImageMagick is available
def check_imagemagick():
    """Check if ImageMagick is installed and available"""
    try:
        from moviepy.config import get_setting, change_settings
        imagemagick_path = get_setting("IMAGEMAGICK_BINARY")
        
        # Check if ImageMagick is installed but path is not configured properly
        if imagemagick_path is None:
            # Try to find ImageMagick in common locations
            common_paths = [
                "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
                "C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\magick.exe",
                "C:\\Program Files\\ImageMagick-7.0.11-Q16-HDRI\\magick.exe",
                "C:\\Program Files\\ImageMagick-7.0.11-Q16\\magick.exe",
                "C:\\Program Files (x86)\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
                "C:\\Program Files (x86)\\ImageMagick-7.1.0-Q16-HDRI\\magick.exe",
                "C:\\Program Files (x86)\\ImageMagick-7.0.11-Q16-HDRI\\magick.exe",
                "C:\\Program Files (x86)\\ImageMagick-7.0.11-Q16\\magick.exe",
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    print(f"Found ImageMagick at {path}")
                    # Update MoviePy settings
                    change_settings({"IMAGEMAGICK_BINARY": path})
                    return True
            
            print("ImageMagick is not installed or not found in common locations.")
            print("To enable text overlays, please install ImageMagick from:")
            print("https://imagemagick.org/script/download.php")
            print("Make sure to check 'Install legacy utilities (convert)' during installation.")
            return False
        
        if not os.path.exists(imagemagick_path):
            print(f"ImageMagick path is configured but file not found: {imagemagick_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking ImageMagick: {e}")
        return False

# Create simple clip without text overlays (fallback when ImageMagick is missing)
def create_simple_clip(clip):
    """Create a simple clip with fade effects but no text overlays"""
    return clip.fadein(1).fadeout(1)

# Add titles and transitions to video clips
def create_titled_clip(clip, title, description):
    # Check if ImageMagick is available
    imagemagick_available = check_imagemagick()
    if not imagemagick_available:
        print("ImageMagick not available, creating clips without text overlays")
        return create_simple_clip(clip)
    
    duration = clip.duration
    
    # Use a more universally available font
    font = 'Arial'
    try:
        # Create title text clip
        txt_title = TextClip(title, fontsize=40, color='white', font=font, 
                             size=clip.size, method='caption', align='center')
        txt_title = txt_title.set_position('center').set_duration(3)
        
        # Create description text clip
        txt_desc = TextClip(description, fontsize=30, color='white', font=font, 
                            size=clip.size, method='caption', align='center')
        txt_desc = txt_desc.set_position('center').set_duration(3).set_start(3)
    except Exception as e:
        print(f"Warning: {e}")
        print("Falling back to default font")
        try:
            # Try again with default font
            txt_title = TextClip(title, fontsize=40, color='white', 
                             size=clip.size, method='caption', align='center')
            txt_title = txt_title.set_position('center').set_duration(3)
            
            txt_desc = TextClip(description, fontsize=30, color='white', 
                            size=clip.size, method='caption', align='center')
            txt_desc = txt_desc.set_position('center').set_duration(3).set_start(3)
        except Exception as e:
            print(f"Text overlay creation failed: {e}")
            return create_simple_clip(clip)
    
    # Create fade in for the title section
    try:
        title_section = CompositeVideoClip([
            clip.subclip(0, min(6, duration)).fadein(1),
            txt_title,
            txt_desc
        ])
        
        # Rest of the clip after title
        if duration > 6:
            main_section = clip.subclip(6, duration)
            final_clip = concatenate_videoclips([title_section, main_section])
        else:
            final_clip = title_section
        
        return final_clip.fadeout(1)
    except Exception as e:
        print(f"Composite clip creation failed: {e}")
        return create_simple_clip(clip)

# Create the story-style video edit
def edit_story_video(original_video_path, clip_data, output_video_path):
    print(f"Creating story video with {len(clip_data['clips'])} clips")
    
    video = VideoFileClip(original_video_path)
    clips = []
    
    # Process each clip
    for i, clip_info in enumerate(clip_data['clips']):
        start = clip_info['start']
        end = clip_info['end']
        title = clip_info['title']
        description = clip_info['description']
        
        print(f"Processing clip {i+1}: {title} ({start}s to {end}s)")
        
        # Extract the clip and add titles
        raw_clip = video.subclip(start, end)
        
        try:
            titled_clip = create_titled_clip(raw_clip, title, description)
            clips.append(titled_clip)
        except Exception as e:
            print(f"Error creating titled clip: {e}")
            print("Adding simple clip without titles")
            # Fallback to simple clip
            simple_clip = create_simple_clip(raw_clip)
            clips.append(simple_clip)
    
    # Create a final clip with the narrative structure
    if clips:
        try:
            # Check if ImageMagick is available for text overlays
            imagemagick_available = check_imagemagick()
            
            # Create the final video based on ImageMagick availability
            if imagemagick_available:
                try:
                    # Create a title for the overall story
                    story_title = "A Short Story"
                    story_desc = clip_data.get('narrative_structure', "Based on your query")
                    
                    # Create intro and outro clips
                    try:
                        intro_clip = TextClip(story_title, fontsize=60, color='white', font='Arial',
                                            size=video.size, bg_color='black', method='caption', align='center')
                        intro_clip = intro_clip.set_duration(5)
                        
                        intro_desc = TextClip(story_desc, fontsize=30, color='white', font='Arial',
                                            size=video.size, bg_color='black', method='caption', align='center')
                        intro_desc = intro_desc.set_position('center').set_duration(5).set_start(5)
                        
                        # Create an outro clip
                        outro_text = "The End"
                        outro_clip = TextClip(outro_text, fontsize=60, color='white', font='Arial',
                                            size=video.size, bg_color='black', method='caption', align='center')
                        outro_clip = outro_clip.set_duration(3)
                        
                        # Combine intro and outro with content
                        intro_composite = CompositeVideoClip([intro_clip, intro_desc]).set_duration(10).fadein(1).fadeout(1)
                        all_clips = [intro_composite] + clips + [outro_clip.fadein(1)]
                    except Exception as e:
                        print(f"Error creating intro/outro: {e}")
                        # Skip intro/outro
                        all_clips = clips
                except Exception as e:
                    print(f"Error with text overlays: {e}")
                    all_clips = clips
            else:
                # Simple concatenation without text overlays
                print("ImageMagick not available, creating simple video without intro/outro")
                all_clips = clips
            
            # Create the final video
            final_clip = concatenate_videoclips(all_clips, method="compose")
            
            # Fix output path
            output_dir = os.path.dirname(output_video_path)
            basename = os.path.basename(output_video_path)
            clean_basename = re.sub(r'[^\w\-\.]', '_', basename)
            
            if clean_basename != basename:
                print(f"Sanitizing output filename from '{basename}' to '{clean_basename}'")
                output_video_path = os.path.join(output_dir, clean_basename)
            
            print(f"Creating final story video at: {output_video_path}")
            final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", 
                                      ffmpeg_params=["-pix_fmt", "yuv420p", "-strict", "-2"])
            
            final_clip.close()
            print("Story video created successfully!")
            
        except Exception as e:
            print(f"Error creating story video: {e}")
            try:
                # Last resort: just concatenate clips without any effects
                print("Attempting basic video concatenation without effects...")
                basic_clips = []
                for clip_info in clip_data['clips']:
                    basic_clip = video.subclip(clip_info['start'], clip_info['end'])
                    basic_clips.append(basic_clip)
                
                if basic_clips:
                    basic_video = concatenate_videoclips(basic_clips)
                    basic_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac",
                                              ffmpeg_params=["-pix_fmt", "yuv420p", "-strict", "-2"])
                    basic_video.close()
                    print("Basic video created successfully!")
            except Exception as e2:
                print(f"Critical error creating video: {e2}")
                import traceback
                traceback.print_exc()
    else:
        print("No clips to include in the story video.")
    
    # Close the original video file
    video.close()

# Define LangChain tools
tools = [
    Tool(
        name="analyze_transcript",
        func=TranscriptAnalysisTool()._run,
        description="Analyzes video transcript and returns segments to include in a video"
    )
]

# Create agent with LangChain
def create_agent():
    # Define the agent prompt
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI video editing assistant. 
        Your goal is to help users create compelling video clips from longer videos.
        You can analyze transcripts, identify relevant segments, and create a story-like structure.
        
        Always follow these steps:
        1. Understand the user's request and confirm the number of clips they want.
        2. Analyze the transcript to find the most relevant and engaging segments.
        3. Structure these segments into a story-like narrative.
        4. Return your findings in a clean, structured format.
        
        Be professional, courteous, and helpful. Explain your choices clearly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent with more explicit configuration
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # Set a reasonable limit on iterations
    )
    
    return agent_executor

def direct_analysis_fallback(transcript, user_query, num_clips):
    """Fallback method that directly calls the LLM without using the agent framework."""
    print("Using direct LLM analysis fallback...")
    analysis_tool = TranscriptAnalysisTool()
    return analysis_tool._run(transcript, user_query, num_clips)

# Main Function
def main():
    # Setup input and output folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_videos")
    output_dir = ensure_output_dir()
    
    # Initialize chat history
    chat_history = []
    
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
    clean_filename = re.sub(r'[^\w\-\.]', '_', f"story_{selected_video}")
    output_video = os.path.abspath(os.path.join(output_dir, clean_filename))
    
    print(f"Selected: {selected_video}")
    
    # User Query
    default_query = "Create an engaging story from this video"
    user_query = input(f"Enter your search query or press Enter for default: ") or default_query
    
    # Number of clips
    num_clips = 3  # Default
    try:
        num_clips_input = input(f"How many clips would you like to include? (default: {num_clips}): ")
        if num_clips_input:
            num_clips = int(num_clips_input)
            if num_clips < 1:
                print("Number of clips must be at least 1. Using default.")
                num_clips = 3
    except ValueError:
        print("Invalid input. Using default number of clips.")
    
    print(f"Using query: {user_query}")
    print(f"Number of clips: {num_clips}")
    print(f"Input video: {input_video}")
    print(f"Output will be saved to: {output_video}")
    
    # Step 1: Transcribe
    print("Transcribing video...")
    transcription = transcribe_video(input_video, model_name="base")
    
    if transcription is None:
        print("Transcription failed. Please check the FFmpeg installation and try again.")
        return
    
    print("Transcription completed successfully.")
    
    # Initialize the agent
    agent_executor = create_agent()
    
    # Prepare user query for agent
    agent_query = f"I want to create {num_clips} clips from a video about: {user_query}. Make it into a short story with a beginning, middle, and end."
    
    # Add transcription success to chat history
    chat_history.append(HumanMessage(content="I've transcribed a video and need help creating clips from it."))
    chat_history.append(AIMessage(content="Great! I can help you analyze the transcript and create engaging video clips. What would you like the clips to be about?"))
    
    # Run the agent
    print("Analyzing transcript with AI agent...")
    clip_data = None
    
    # Try the agent approach first
    try:
        # Attempt to run the agent to analyze the transcript
        try:
            agent_response = agent_executor.invoke({
                "input": agent_query,
                "chat_history": chat_history,
                "transcript": transcription,
                "num_clips": num_clips
            })
            
            print("AI Agent Response:")
            print(agent_response.get("output", "No output from agent"))
            
            # Extract clips from agent response
            try:
                intermediate_steps = agent_response.get("intermediate_steps", [])
                if intermediate_steps:
                    for step in intermediate_steps:
                        if isinstance(step[1], dict) and "clips" in step[1]:
                            clip_data = step[1]
                            break
            except (AttributeError, IndexError, TypeError) as e:
                print(f"Could not extract clips from intermediate steps: {e}")
        except Exception as e:
            print(f"Agent execution failed: {e}")
            print("Continuing with direct transcript analysis...")
    except Exception as e:
        print(f"Error in agent processing: {e}")
    
    # If the agent approach failed, fall back to direct analysis
    if not clip_data:
        print("Agent approach unsuccessful, using fallback method...")
        clip_data = direct_analysis_fallback(transcription, user_query, num_clips)
    
    # Final fallback in case all else fails
    if not clip_data or not clip_data.get("clips"):
        print("All approaches failed. Creating basic clips...")
        # Create simple clips evenly spaced throughout the video
        clips = []
        total_duration = transcription[-1]["end"] - transcription[0]["start"]
        clip_duration = min(60, total_duration / num_clips)
        
        for i in range(num_clips):
            start_time = transcription[0]["start"] + (i * clip_duration)
            end_time = min(transcription[-1]["end"], start_time + clip_duration)
            
            clips.append({
                "start": start_time,
                "end": end_time,
                "title": f"Clip {i+1}",
                "description": f"Auto-generated clip {i+1}"
            })
        
        clip_data = {
            "clips": clips,
            "narrative_structure": "Automatically generated clip sequence"
        }
    
    # Step 3: Edit Video
    print(f"Creating story video with {len(clip_data['clips'])} clips...")
    print(f"Narrative structure: {clip_data.get('narrative_structure', 'Not provided')}")
    
    edit_story_video(input_video, clip_data, output_video)
    print(f"Story video creation complete. Saved to: {output_video}")

if __name__ == "__main__":
    main() 