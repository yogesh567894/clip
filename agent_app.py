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

# Import emotion analysis capabilities
try:
    from audio_emotion_analyzer import AudioEmotionAnalyzer
    from emotion_agent import process_video_emotions
    EMOTION_ANALYSIS_AVAILABLE = True
    print("Emotion analysis capabilities loaded successfully")
except ImportError as e:
    print(f"Emotion analysis not available: {e}")
    EMOTION_ANALYSIS_AVAILABLE = False

# NVIDIA API key
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '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')

# Configuration for token limits and optimization
CONFIG = {
    "token_limits": {
        "max_tokens_per_segment": 4000,  # Conservative limit for each segment
        "transcript_token_threshold": 7000,  # When to trigger chunking
        "token_safety_margin": 1000,  # Safety margin for prompts
    },
    "clip_limits": {
        "min_duration": 20,  # Minimum clip duration in seconds
        "max_duration": 90,  # Maximum clip duration in seconds
        "optimal_duration": 45,  # Target clip duration in seconds
        "max_clips_per_chunk": 3  # Maximum clips to extract from a single chunk
    }
}

# Initialize the LLM
try:
    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct", 
        api_key=NVIDIA_API_KEY,
        streaming=False  # Disable streaming to avoid the chunk error
    )
    print("Successfully initialized NVIDIA AI")
except Exception as e:
    print(f"Error initializing NVIDIA AI: {e}")
    print("Using fallback mechanism for transcript analysis")
    llm = None  # We'll use a direct approach without LLM if it fails

# Initialize emotion analyzer if available
emotion_analyzer = None
if EMOTION_ANALYSIS_AVAILABLE:
    try:
        emotion_analyzer = AudioEmotionAnalyzer()
        print("Audio emotion analyzer initialized")
    except Exception as e:
        print(f"Error initializing emotion analyzer: {e}")
        EMOTION_ANALYSIS_AVAILABLE = False

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

# Tool for emotion-based highlight extraction (new)
class EmotionHighlightTool(BaseTool):
    name = "extract_emotional_highlights"
    description = "Extracts video clips based on emotional content in the audio"
    
    def _run(self, video_path: str, emotion_type: Optional[str] = None, num_clips: int = 3) -> Dict[str, Any]:
        if not EMOTION_ANALYSIS_AVAILABLE or not emotion_analyzer:
            return {"error": "Emotion analysis is not available"}
        
        print(f"Extracting {num_clips} emotional highlights (emotion: {emotion_type or 'any'})")
        
        try:
            # Use the emotion analyzer to get highlights
            highlights = emotion_analyzer.get_emotional_highlights(
                video_path,
                emotion_preference=emotion_type,
                num_clips=num_clips
            )
            
            # Return the results
            return {
                "clips": highlights,
                "narrative_structure": f"Emotional highlights focused on {emotion_type or 'strong emotions'}"
            }
        except Exception as e:
            print(f"Error in emotion highlight extraction: {e}")
            return {"error": f"Failed to extract emotional highlights: {str(e)}"}

# Tool for hybrid content-emotion analysis (new)
class HybridAnalysisTool(BaseTool):
    name = "hybrid_content_emotion_analysis"
    description = "Analyzes both transcript content and emotional signals to find the best clips"
    
    def _run(self, transcript: List[Dict], video_path: str, user_query: str, num_clips: int = 3) -> Dict[str, Any]:
        print(f"Performing hybrid content-emotion analysis for {num_clips} clips")
        
        # First get content-based clips using the transcript
        content_tool = TranscriptAnalysisTool()
        content_results = content_tool._run(transcript, user_query, num_clips)
        
        # Check if content analysis was successful
        if not content_results or not content_results.get("clips") or "error" in content_results:
            print("Content analysis failed, falling back to emotion-only analysis")
            # Fall back to emotion-only analysis
            if EMOTION_ANALYSIS_AVAILABLE and emotion_analyzer:
                emotion_highlights = emotion_analyzer.get_emotional_highlights(
                    video_path,
                    num_clips=num_clips
                )
                return {
                    "clips": emotion_highlights,
                    "narrative_structure": "Emotion-based highlights (content analysis failed)"
                }
            else:
                return {"error": "Both content and emotion analysis failed"}
        
        # If emotion analysis is not available, return content results only
        if not EMOTION_ANALYSIS_AVAILABLE or not emotion_analyzer:
            return content_results
        
        # Get emotion data for the content-based clips
        try:
            content_clips = content_results["clips"]
            enriched_clips = emotion_analyzer.analyze_video_segments(video_path, content_clips)
            
            # Update clips with emotion data
            for i, clip in enumerate(enriched_clips):
                if "error" not in clip and "emotion_analysis" in clip:
                    emotion_data = clip["emotion_analysis"]
                    content_clips[i]["dominant_emotion"] = emotion_data.get("dominant_emotion", "unknown")
                    content_clips[i]["emotion_scores"] = emotion_data.get("emotion_scores", {})
                    
                    # Update title to include emotion
                    emotion = content_clips[i]["dominant_emotion"].capitalize()
                    if "title" in content_clips[i]:
                        content_clips[i]["title"] = f"{emotion}: {content_clips[i]['title']}"
                    
                    # Update description to include emotion
                    if "description" in content_clips[i]:
                        content_clips[i]["description"] += f" with {emotion} emotional tone"
            
            return {
                "clips": content_clips,
                "narrative_structure": content_results.get("narrative_structure", "") + " (enriched with emotion analysis)"
            }
            
        except Exception as e:
            print(f"Error in hybrid analysis: {e}")
            # Return the original content results if emotion enrichment fails
            return content_results

# Add this function to check for emotion-related queries
def is_emotion_focused_query(query):
    """Check if a query is focused on emotional content"""
    emotion_keywords = [
        "emotion", "emotional", "feel", "feeling", "mood", "happy", "sad", 
        "angry", "excited", "surprising", "surprised", "fear", "fearful",
        "disgust", "calm", "neutral", "passionate", "enthusiasm", "enthusiastic",
        "joy", "joyful", "sorrow", "sorrowful", "rage", "outrage", "laugh", "crying"
    ]
    
    query_lower = query.lower()
    for keyword in emotion_keywords:
        if keyword in query_lower:
            return True
    
    return False

# Extract specific emotion from query
def extract_emotion_from_query(query):
    """Extract specific emotion type from a query"""
    emotion_mapping = {
        "happy": ["happy", "joy", "joyful", "happiness", "exciting", "excited", "positive"],
        "sad": ["sad", "sorrow", "sorrowful", "sadness", "depressing", "gloomy"],
        "angry": ["angry", "anger", "rage", "outrage", "furious", "fury", "mad"],
        "fearful": ["fear", "fearful", "scared", "scary", "terrified", "terror", "afraid"],
        "surprised": ["surprise", "surprised", "surprising", "amazed", "shocked", "astonished"],
        "disgust": ["disgust", "disgusted", "disgusting", "repulsed"],
        "calm": ["calm", "peaceful", "relaxed", "relaxing", "soothing"],
        "neutral": ["neutral", "balanced", "objective", "factual"]
    }
    
    query_lower = query.lower()
    for emotion, keywords in emotion_mapping.items():
        for keyword in keywords:
            if keyword in query_lower:
                return emotion
    
    return None

# Tool for analyzing transcript and generating clip segments
class TranscriptAnalysisTool(BaseTool):
    name = "analyze_transcript"
    description = "Analyzes video transcript and returns segments to include in a video"
    
    def _run(self, transcript: List[Dict], user_query: str, num_clips: int = 1) -> Dict[str, Any]:
        print(f"Analyzing transcript for {num_clips} clips based on query: '{user_query}'")
        
        # Estimate token count for the transcript
        transcript_json = json.dumps(transcript)
        estimated_tokens = estimate_tokens(transcript_json)
        threshold = CONFIG["token_limits"]["transcript_token_threshold"]
        
        print(f"Estimated transcript tokens: {estimated_tokens} (threshold: {threshold})")
        
        # Adapt num_clips based on transcript length and save original request
        original_num_clips = num_clips
        
        if num_clips > 1 and estimated_tokens > threshold:
            # Get optimal clip count for this transcript size
            adjusted_clips = get_optimal_clip_count(num_clips, estimated_tokens, threshold)
            
            if adjusted_clips != num_clips:
                print(f"Adjusting requested clips from {num_clips} to {adjusted_clips} due to transcript size")
                num_clips = adjusted_clips
        
        # Check if transcript is too large for the context window
        if estimated_tokens > threshold:
            print(f"Transcript is too large ({estimated_tokens} estimated tokens), using chunking approach...")
            return self._analyze_large_transcript(transcript, user_query, num_clips, original_num_clips)
        
        # Continue with normal processing for reasonably sized transcripts
        prompt = f"""You are an expert video editor who can read video transcripts and identify the most important segments. Given a transcript with timestamps, identify exactly {num_clips} compelling segments that would make excellent clips related to the user's query.

Guidelines:
1. Identify exactly {num_clips} distinct segments that are MOST relevant to the user query.
2. If more than one clip is requested, PRIORITIZE including both an introduction and conclusion segment.
3. The introduction is typically found in the first 15% of the video and often contains phrases like "welcome", "introduction", "today we'll talk about", etc.
4. The conclusion is typically found in the last 15% of the video and often contains phrases like "in conclusion", "summary", "thank you", "subscribe", etc.
5. Each clip should be {CONFIG["clip_limits"]["min_duration"]}-{CONFIG["clip_limits"]["max_duration"]} seconds in duration, but prioritize coherence over exact duration.
6. Be extremely precise about start and end timestamps - cut at natural pauses.
7. The middle segments should contain the most important information related to the query.

Output format (IMPORTANT): You must respond ONLY with valid JSON in this format:
{{
  "clips": [
    {{
      "start": start_time_in_seconds,
      "end": end_time_in_seconds,
      "title": "Introduction: Brief descriptive title",
      "description": "Short description of why this segment is important"
    }},
    ...additional clips...
  ],
  "narrative_structure": "Brief explanation of why these specific clips were chosen"
}}

For the titles, use prefixes like "Introduction:", "Conclusion:", or "Highlight:" based on the segment position and content.

Transcript:
{transcript_json}

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
                    
                    # Enforce clip duration limits
                    duration = end - start
                    if duration < CONFIG["clip_limits"]["min_duration"]:
                        # Try to extend the clip to minimum duration
                        end = min(transcript[-1]["end"], start + CONFIG["clip_limits"]["min_duration"])
                    elif duration > CONFIG["clip_limits"]["max_duration"]:
                        # Limit the clip to maximum duration
                        end = start + CONFIG["clip_limits"]["max_duration"]
                    
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
                    total_duration = transcript[-1]["end"] - transcript[0]["start"]
                    clip_duration = min(CONFIG["clip_limits"]["optimal_duration"], 
                                       total_duration / num_clips)
                    
                    for i in range(num_clips):
                        start_time = transcript[0]["start"] + (i * total_duration / num_clips)
                        
                        # Find closest segment
                        closest_segment = min(transcript, key=lambda x: abs(x["start"] - start_time))
                        start_time = closest_segment["start"]
                        end_time = min(transcript[-1]["end"], start_time + clip_duration)
                        
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
                # Fallback to heuristic approach
                return analyze_transcript_without_llm(transcript, user_query, num_clips)
                
        except Exception as e:
            print(f"Error in transcript analysis: {e}")
            # For token limit errors, try the chunking approach
            if "maximum context length" in str(e) or "token limit" in str(e).lower():
                print("Token limit exceeded, using chunking approach...")
                return self._analyze_large_transcript(transcript, user_query, num_clips, original_num_clips)
            
            # Use heuristic approach as final fallback
            return analyze_transcript_without_llm(transcript, user_query, num_clips)
    
    def _analyze_large_transcript(self, transcript: List[Dict], user_query: str, num_clips: int, original_requested_clips: int = None) -> Dict[str, Any]:
        """Handle large transcripts by analyzing them in chunks based on tokens"""
        print("Using chunked transcript analysis approach...")
        original_requested_clips = original_requested_clips or num_clips
        
        # First identify the total duration of the video
        total_duration = transcript[-1]["end"] - transcript[0]["start"]
        print(f"Total video duration: {total_duration:.2f} seconds")
        
        # Determine chunking strategy based on clip count and transcript size
        if num_clips <= 3:
            # For small number of clips, we'll use semantic chunking (intro, middle, end)
            # Adjust the intro and ending thresholds based on video length
            intro_percentage = min(0.15, max(0.05, 5 * 60 / total_duration))  # 5 minutes or 5-15%
            ending_percentage = min(0.15, max(0.05, 5 * 60 / total_duration))
            
            print(f"Using semantic chunking with intro threshold: {intro_percentage:.1%}, ending threshold: {ending_percentage:.1%}")
            
            # Define the sections
            intro_cutoff = transcript[0]["start"] + (total_duration * intro_percentage)
            ending_start = transcript[0]["start"] + (total_duration * (1 - ending_percentage))
            
            # Divide transcript into sections
            intro_section = []
            middle_section = []
            ending_section = []
            
            for segment in transcript:
                if segment["start"] < intro_cutoff:
                    intro_section.append(segment)
                elif segment["start"] > ending_start:
                    ending_section.append(segment)
                else:
                    middle_section.append(segment)
            
            print(f"Divided transcript: intro ({len(intro_section)} segments), middle ({len(middle_section)} segments), ending ({len(ending_section)} segments)")
            
            # Get clips based on the requested number
            clips = []
            
            if num_clips == 1:
                # With 1 clip, focus on the most relevant section
                if "intro" in user_query.lower() or "beginning" in user_query.lower():
                    clips = self._get_best_clips_from_section(intro_section, user_query, 1)
                elif "end" in user_query.lower() or "conclusion" in user_query.lower():
                    clips = self._get_best_clips_from_section(ending_section, user_query, 1)
                else:
                    # Try middle section first, then use heuristic approach
                    middle_clips = self._get_best_clips_from_section(middle_section, user_query, 1)
                    if middle_clips:
                        clips = middle_clips
                    else:
                        return analyze_transcript_without_llm(transcript, user_query, num_clips)
            elif num_clips == 2:
                # With 2 clips, get intro and ending
                intro_clip = self._get_best_clips_from_section(intro_section, "introduction", 1)
                ending_clip = self._get_best_clips_from_section(ending_section, "conclusion", 1)
                clips = intro_clip + ending_clip
            else:  # num_clips == 3
                # With 3 clips, get intro, middle, and ending
                intro_clip = self._get_best_clips_from_section(intro_section, "introduction", 1)
                middle_clip = self._get_best_clips_from_section(middle_section, user_query, 1)
                ending_clip = self._get_best_clips_from_section(ending_section, "conclusion", 1)
                clips = intro_clip + middle_clip + ending_clip
        else:
            # For more clips, use token-based chunking
            print(f"Using token-based chunking for {num_clips} clips")
            
            # Create chunks based on token limits
            target_token_limit = CONFIG["token_limits"]["max_tokens_per_segment"]
            chunks = chunk_transcript_by_tokens(transcript, target_token_limit)
            
            print(f"Split transcript into {len(chunks)} token-based chunks")
            
            # Calculate clips per chunk
            if len(chunks) >= num_clips:
                # If we have enough chunks, get 1 clip from the best chunks
                clips_needed = num_clips
                chunks_to_use = clips_needed
                clips_per_chunk = 1
            else:
                # Distribute clips among chunks
                clips_per_chunk = num_clips // len(chunks)
                remainder = num_clips % len(chunks)
                chunks_to_use = len(chunks)
                
                # If very few chunks but many clips requested, limit clips per chunk
                max_per_chunk = CONFIG["clip_limits"]["max_clips_per_chunk"]
                if clips_per_chunk > max_per_chunk:
                    clips_per_chunk = max_per_chunk
                    chunks_to_use = min(len(chunks), num_clips // clips_per_chunk + (1 if remainder > 0 else 0))
                    remainder = num_clips - (chunks_to_use * clips_per_chunk)
            
            print(f"Using {chunks_to_use} chunks with ~{clips_per_chunk} clips per chunk (remainder: {remainder})")
            
            # Score chunks for relevance to determine which to use and for extra clips
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                # Basic ranking factors
                chunk_position = i / len(chunks)
                is_intro = chunk_position < 0.2  # First 20% of chunks
                is_ending = chunk_position > 0.8  # Last 20% of chunks
                
                # Start with base score from position
                score = 0
                
                # Boost intro and ending chunks
                if is_intro:
                    score += 10
                    # Extra boost for first chunk
                    if i == 0:
                        score += 5
                if is_ending:
                    score += 10
                    # Extra boost for last chunk
                    if i == len(chunks) - 1:
                        score += 5
                
                # Score middle chunks based on the query
                query_keywords = [kw.lower() for kw in user_query.split() if len(kw) > 3]
                for segment in chunk:
                    text = segment["text"].lower()
                    # Count keyword matches
                    for keyword in query_keywords:
                        if keyword in text:
                            score += 5
                
                chunk_scores.append((i, score))
            
            # Sort chunks by score (best first)
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top chunks based on score
            top_chunk_indices = [idx for idx, _ in chunk_scores[:chunks_to_use]]
            
            # Determine which chunks get extra clips from remainder
            extra_clip_indices = [idx for idx, _ in chunk_scores[:remainder]]
            
            # Get clips from each selected chunk
            clips = []
            
            # Process chunks in chronological order for better narrative flow
            top_chunk_indices.sort()
            
            for i in top_chunk_indices:
                chunk = chunks[i]
                # Determine clips to get from this chunk
                this_chunk_clips = clips_per_chunk
                if i in extra_clip_indices:
                    this_chunk_clips += 1
                
                # Choose appropriate query based on chunk position
                chunk_position = i / len(chunks)
                if chunk_position < 0.2:  # Early in video
                    query = "introduction" if "intro" in user_query.lower() else user_query
                elif chunk_position > 0.8:  # Late in video
                    query = "conclusion" if "conclusion" in user_query.lower() else user_query
                else:  # Middle of video
                    query = user_query
                
                # Get clips from this chunk
                chunk_clips = self._get_best_clips_from_section(chunk, query, this_chunk_clips)
                clips.extend(chunk_clips)
        
        # Ensure we have the requested number of clips
        if len(clips) < num_clips:
            print(f"Only found {len(clips)} clips, using heuristic approach for remaining {num_clips - len(clips)}")
            remaining_clips = analyze_transcript_without_llm(transcript, user_query, num_clips - len(clips))
            clips.extend(remaining_clips["clips"])
        
        # Sort clips by start time
        clips.sort(key=lambda x: x["start"])
        
        # Deduplicate clips that might overlap significantly
        if len(clips) > 1:
            deduplicated_clips = [clips[0]]
            for i in range(1, len(clips)):
                current = clips[i]
                prev = deduplicated_clips[-1]
                
                # Check for significant overlap (more than 50%)
                overlap_start = max(current["start"], prev["start"])
                overlap_end = min(current["end"], prev["end"])
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    current_duration = current["end"] - current["start"]
                    
                    # If overlap is more than 50% of the current clip
                    if overlap_duration > (current_duration * 0.5):
                        # Skip this clip
                        continue
                
                deduplicated_clips.append(current)
            
            clips = deduplicated_clips
        
        # If after deduplication we have fewer clips than requested, add more
        if len(clips) < num_clips:
            print(f"After deduplication, only have {len(clips)} clips, adding more to reach {num_clips}")
            # Find gaps between existing clips to add more
            remaining_needed = num_clips - len(clips)
            existing_start_times = [clip["start"] for clip in clips]
            existing_end_times = [clip["end"] for clip in clips]
            
            # Create gap segments where we can add new clips
            gaps = []
            sorted_clips = sorted(clips, key=lambda x: x["start"])
            
            # Add initial gap if needed
            if sorted_clips and sorted_clips[0]["start"] > transcript[0]["start"] + 30:
                gaps.append((transcript[0]["start"], sorted_clips[0]["start"] - 10))
            
            # Add middle gaps
            for i in range(len(sorted_clips) - 1):
                curr_end = sorted_clips[i]["end"]
                next_start = sorted_clips[i+1]["start"]
                if next_start - curr_end > 60:  # Only use significant gaps (>60s)
                    gaps.append((curr_end + 10, next_start - 10))
            
            # Add final gap if needed
            if sorted_clips and sorted_clips[-1]["end"] < transcript[-1]["end"] - 30:
                gaps.append((sorted_clips[-1]["end"] + 10, transcript[-1]["end"]))
            
            # Fill gaps with new clips if we have enough gaps
            if gaps and len(gaps) >= remaining_needed:
                gap_clips = []
                for i, (gap_start, gap_end) in enumerate(gaps[:remaining_needed]):
                    # Find segments in this gap
                    gap_segments = [s for s in transcript if gap_start <= s["start"] < gap_end]
                    if gap_segments:
                        gap_clips.extend(self._get_best_clips_from_section(gap_segments, user_query, 1))
                
                clips.extend(gap_clips)
                # Re-sort by start time
                clips.sort(key=lambda x: x["start"])
            else:
                # Not enough gaps, use heuristic approach
                more_clips = analyze_transcript_without_llm(transcript, user_query, remaining_needed)
                clips.extend(more_clips["clips"])
                # Re-sort by start time
                clips.sort(key=lambda x: x["start"])
                
                return {
            "clips": clips[:num_clips],  # Limit to requested number
            "narrative_structure": "Clips selected from chunked transcript analysis"
        }
    
    def _get_best_clips_from_section(self, section: List[Dict], query: str, num_clips: int) -> List[Dict]:
        """Analyze a section of transcript to find the best clips"""
        if not section:
            return []
        
        try:
            # Try to use LLM for small sections
            section_json = json.dumps(section)
            estimated_tokens = estimate_tokens(section_json)
            max_segment_tokens = CONFIG["token_limits"]["max_tokens_per_segment"]
            
            if estimated_tokens < max_segment_tokens:
                print(f"Using LLM for section analysis ({estimated_tokens} estimated tokens)")
                prompt = f"""Analyze this transcript section and identify {num_clips} segments that would make good clips about "{query}".
                
Each clip should be {CONFIG["clip_limits"]["min_duration"]}-{CONFIG["clip_limits"]["max_duration"]} seconds in duration and contain complete thoughts.
                
Respond ONLY with JSON in this format:
{{
  "clips": [
    {{
      "start": start_time_in_seconds,
      "end": end_time_in_seconds,
      "title": "Brief descriptive title",
      "description": "Why this segment is important"
    }}
  ]
}}

Transcript section:
{section_json}"""

                response = llm.invoke(prompt)
                
                # Parse the response
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                
                # Extract JSON
                json_match = re.search(r'(\{.*"clips":\s*\[.*\].*\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        json_str = content
                
                # Clean and parse JSON
                cleaned_json = json_str.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(cleaned_json)
                    if "clips" in result and result["clips"]:
                        clips = []
                        for clip in result["clips"][:num_clips]:
                            # Ensure required fields with proper types
                            start = float(clip["start"]) if isinstance(clip["start"], str) else clip["start"]
                            end = float(clip["end"]) if isinstance(clip["end"], str) else clip["end"]
                            
                            # Validate clip
                            if end > start:
                                clips.append({
                                    "start": start,
                                    "end": end,
                                    "title": clip.get("title", f"Clip from {start:.1f}s to {end:.1f}s"),
                                    "description": clip.get("description", "Relevant segment")
                                })
                        
                        if clips:
                            return clips
                except Exception as e:
                    print(f"Error parsing section JSON: {e}")
                    print(f"Error in section analysis: {e}")
        
        except Exception as e:
            print(f"Error in section analysis: {e}")
        
        # Fallback to simpler heuristic approach for this section
        return self._analyze_section_heuristically(section, query, num_clips)
    
    def _analyze_section_heuristically(self, section: List[Dict], query: str, num_clips: int) -> List[Dict]:
        """Use simple heuristics to find good clips in a section"""
        if not section:
            return []
        
        print(f"Using heuristic analysis for section with {len(section)} segments")
        
        # Generate clips evenly spaced throughout the section
        clips = []
        min_duration = CONFIG["clip_limits"]["min_duration"]
        max_duration = CONFIG["clip_limits"]["max_duration"]
        optimal_duration = CONFIG["clip_limits"]["optimal_duration"]
        
        if len(section) <= num_clips:
            # If we have fewer segments than clips, use each segment
            for i, segment in enumerate(section):
                # Extend segment duration if possible
                start = segment["start"]
                end = segment["end"]
                
                # Try to extend to about optimal duration
                if i < len(section) - 1:
                    next_end = min(section[i+1]["end"], start + optimal_duration)
                    end = max(end, next_end)
                
                # Ensure minimum duration
                if end - start < min_duration and i < len(section) - 1:
                    end = min(section[-1]["end"], start + min_duration)
                
                # Limit to maximum duration
                if end - start > max_duration:
                    end = start + max_duration
                    
                # Create clip title based on position in video
                title = f"Clip {i+1}"
                if "introduction" in query.lower():
                    title = f"Introduction: {extract_topic(segment['text'])}"
                elif "conclusion" in query.lower():
                    title = f"Conclusion: {extract_topic(segment['text'])}"
                else:
                    title = f"Highlight: {extract_topic(segment['text'])}"
                
                clips.append({
                    "start": start,
                    "end": end,
                    "title": title,
                    "description": f"Segment containing content about {query}"
                })
        else:
            # Calculate segment duration
            section_duration = section[-1]["end"] - section[0]["start"]
            clip_interval = section_duration / num_clips
            
            for i in range(num_clips):
                target_time = section[0]["start"] + (i * clip_interval)
                
                # Find closest segment
                closest_idx = min(range(len(section)), 
                                key=lambda j: abs(section[j]["start"] - target_time))
                
                segment = section[closest_idx]
                start = segment["start"]
                
                # Try to create a 30-60 second clip around this segment
                end = min(section[-1]["end"], start + 60)
                
                # Create clip title based on position in video
                title = f"Clip {i+1}"
                if "introduction" in query.lower():
                    title = f"Introduction: {extract_topic(segment['text'])}"
                elif "conclusion" in query.lower():
                    title = f"Conclusion: {extract_topic(segment['text'])}"
                else:
                    title = f"Highlight: {extract_topic(segment['text'])}"
                    
                clips.append({
                    "start": start, 
                    "end": end,
                    "title": title,
                    "description": f"Segment containing content about {query}"
                })
        
        return clips[:num_clips]  # Ensure we return at most num_clips

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

# Modified function to create individual clips instead of a story video
def create_individual_clips(original_video_path, clip_data, output_dir):
    print(f"Creating {len(clip_data['clips'])} individual clips")
    
    video = VideoFileClip(original_video_path)
    output_paths = []
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(original_video_path))[0]
    base_filename = re.sub(r'[^\w\-\.]', '_', base_filename)
    
    # Process each clip individually
    for i, clip_info in enumerate(clip_data['clips']):
        start = clip_info['start']
        end = clip_info['end']
        title = clip_info['title']
        description = clip_info['description']
        
        print(f"Processing clip {i+1}: {title} ({start}s to {end}s)")
        
        try:
            # Extract the clip
            raw_clip = video.subclip(start, end)
            
            # Add title if ImageMagick is available
            imagemagick_available = check_imagemagick()
            if imagemagick_available:
                titled_clip = create_titled_clip(raw_clip, title, description)
            else:
                print("ImageMagick not available, creating clip without text overlays")
                titled_clip = create_simple_clip(raw_clip)
                
            # Create descriptive filename based on title
            clip_type = "clip"
            if "introduction" in title.lower():
                clip_type = "intro"
            elif "conclusion" in title.lower():
                clip_type = "outro"
            
            # Clean the title for the filename
            title_for_filename = re.sub(r'[^\w\-\.]', '_', title.split(":")[0].strip())
            if len(title_for_filename) > 20:
                title_for_filename = title_for_filename[:20]
                
            # Create unique output path for this clip
            output_filename = f"{base_filename}_{clip_type}_{i+1}_{int(start)}s_to_{int(end)}s_{title_for_filename}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            output_paths.append(output_path)
            
            print(f"Creating clip {i+1} at: {output_path}")
            
            # Fix error by using enhanced error handling for video writing
            try:
                titled_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", 
                                      ffmpeg_params=["-pix_fmt", "yuv420p", "-strict", "-2"])
            except Exception as write_error:
                print(f"Error with standard writer: {write_error}")
                print("Trying alternate encoding method...")
                # Alternative method that might work better on some systems
                try:
                    titled_clip.write_videofile(output_path, codec="libx264", 
                                            ffmpeg_params=["-pix_fmt", "yuv420p"],
                                            verbose=False)
                except Exception as alt_error:
                    print(f"Alternative encoding also failed: {alt_error}")
                    raise
            
            titled_clip.close()
            print(f"Clip {i+1} created successfully!")
            
        except Exception as e:
            print(f"Error creating clip {i+1}: {e}")
            try:
                print("Attempting basic clip creation without effects...")
                simple_clip = video.subclip(start, end)
                simple_path = os.path.join(output_dir, f"{base_filename}_simple_{i+1}_{int(start)}s_to_{int(end)}s.mp4")
                simple_clip.write_videofile(simple_path, codec="libx264", 
                                         verbose=False,
                                         ffmpeg_params=["-pix_fmt", "yuv420p"])
                simple_clip.close()
                output_paths.append(simple_path)
                print(f"Simple clip {i+1} created successfully as fallback!")
            except Exception as fallback_error:
                print(f"Even simple fallback failed: {fallback_error}")
    
    # Close the original video file
    video.close()

    return output_paths

# Modify the agent creation to fix the error
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
    
    # Create the agent with more explicit configuration for NVIDIA AI
    try:
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
            max_iterations=3  # Limit iterations to prevent excessive API calls
        )
        
        return agent_executor
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

# Define LangChain tools
tools = [
    Tool(
        name="analyze_transcript",
        func=TranscriptAnalysisTool()._run,
        description="Analyzes video transcript and returns segments to include in a video"
    )
]

# Helper function to extract topic from segment text
def extract_topic(text):
    """Extract a likely topic from segment text for better titles"""
    # Remove common filler words
    fillers = ["um", "uh", "like", "you know", "so", "anyway", "basically", "actually", "right"]
    cleaned_text = text.lower()
    for filler in fillers:
        cleaned_text = cleaned_text.replace(f" {filler} ", " ")
    
    # Try to find a concise topic
    sentences = cleaned_text.split('.')
    if not sentences:
        return text[:30]
    
    # Use the first sentence if it's short enough
    first_sentence = sentences[0].strip()
    if 10 <= len(first_sentence) <= 50:
        return first_sentence.capitalize()
    
    # Find a key phrase (subject-verb or similar structure)
    words = first_sentence.split()
    if 5 <= len(words) <= 15:
        return ' '.join(words).capitalize()
    
    # Just use the first few words if all else fails
    return ' '.join(words[:8]).capitalize() + "..."

# Function for transcript analysis when LLM is not available
def analyze_transcript_without_llm(transcript, user_query, num_clips=1):
    """Analyze transcript without using LLM - use simple heuristics instead"""
    print("Using heuristic-based transcript analysis")
    
    clips = []
    if not transcript:
        return {"clips": [], "narrative_structure": "No transcript available"}
    
    # Calculate total duration
    total_duration = transcript[-1]["end"] - transcript[0]["start"]
    
    # Intro detection - typically first 2-3 segments
    intro_segments = transcript[:min(3, len(transcript))]
    
    # Ending detection - typically last 2-3 segments
    ending_segments = transcript[max(0, len(transcript)-3):]
    
    # Find segments with potential keywords from the query
    keywords = [kw.lower() for kw in user_query.split() if len(kw) > 3]
    
    # Identify segments containing keywords
    scored_segments = []
    for i, segment in enumerate(transcript):
        segment_text = segment["text"].lower()
        score = 0
        
        # Score based on keywords
        for keyword in keywords:
            if keyword in segment_text:
                score += 10
        
        # Score based on segment length (prefer 30-60 second segments)
        segment_duration = segment["end"] - segment["start"]
        if 30 <= segment_duration <= 60:
            score += 5
        
        # Boost score for introductions (first 15% of video)
        position_percent = segment["start"] / total_duration * 100
        if position_percent < 15:
            score += 15
            if any(intro_indicator in segment_text.lower() for intro_indicator in 
                  ["welcome", "introduction", "today", "hello", "hi everyone", "going to", "talk about"]):
                score += 20
        
        # Boost score for endings (last 15% of video)
        if position_percent > 85:
            score += 15
            if any(outro_indicator in segment_text.lower() for outro_indicator in 
                  ["conclusion", "summary", "thank you", "thanks for", "subscribe", "that's all", "finally"]):
                score += 20
        
        # Save scores
        scored_segments.append({
            "index": i,
            "start": segment["start"],
            "end": segment["end"],
            "score": score,
            "text": segment["text"],
            "is_intro": position_percent < 15,
            "is_ending": position_percent > 85
        })
    
    # Sort by score
    scored_segments.sort(key=lambda x: x["score"], reverse=True)
    
    # Always include an intro if available and we need at least 2 clips
    intro_selected = False
    intro_segment = None
    
    if num_clips >= 2:
        for segment in sorted([s for s in scored_segments if s["is_intro"]], key=lambda x: x["score"], reverse=True):
            intro_segment = segment
            break
    
    # Always include an ending if available and we need at least 2 clips
    ending_selected = False
    ending_segment = None
    
    if num_clips >= 2:
        for segment in sorted([s for s in scored_segments if s["is_ending"]], key=lambda x: x["score"], reverse=True):
            ending_segment = segment
            break
    
    # If we need only 1 clip but have high-scoring intro/ending, use the higher scored one
    if num_clips == 1 and intro_segment and ending_segment:
        if intro_segment["score"] > ending_segment["score"]:
            ending_segment = None
        else:
            intro_segment = None
    
    # Select top segments up to num_clips, ensuring we include intro and ending if available
    remaining_slots = num_clips
    selected_indices = set()
    
    # Add intro if we found one
    if intro_segment:
        # Extend intro to make it more meaningful
        start_idx = max(0, intro_segment["index"] - 1)
        end_idx = min(len(transcript) - 1, intro_segment["index"] + 2)
        
        start_time = transcript[start_idx]["start"]
        end_time = transcript[end_idx]["end"]
        
        # Limit to reasonable duration (max 60 seconds)
        if end_time - start_time > 60:
            end_time = start_time + 60
        
        topic = extract_topic(intro_segment["text"])
        clips.append({
            "start": start_time,
            "end": end_time,
            "title": "Introduction: " + topic,
            "description": "Opening segment of the video"
        })
        
        selected_indices.add(intro_segment["index"])
        intro_selected = True
        remaining_slots -= 1
    
    # Add ending if we found one
    if ending_segment and remaining_slots > 0:
        # Extend ending to capture conclusion
        start_idx = max(0, ending_segment["index"] - 1)
        end_idx = min(len(transcript) - 1, ending_segment["index"] + 1)
        
        start_time = transcript[start_idx]["start"]
        end_time = transcript[end_idx]["end"]
        
        # Limit to reasonable duration (max 60 seconds)
        if end_time - start_time > 60:
            end_time = start_time + 60
        
        topic = extract_topic(ending_segment["text"])
        clips.append({
            "start": start_time,
            "end": end_time,
            "title": "Conclusion: " + topic,
            "description": "Closing segment of the video"
        })
        
        selected_indices.add(ending_segment["index"])
        ending_selected = True
        remaining_slots -= 1
    
    # Fill remaining slots with other high-scoring segments
    for segment in scored_segments:
        if remaining_slots <= 0:
            break
            
        if segment["index"] in selected_indices:
            continue
            
        # Don't select adjacent segments
        adjacent = False
        for idx in selected_indices:
            if abs(segment["index"] - idx) <= 1:
                adjacent = True
                break
                
        if not adjacent:
            selected_indices.add(segment["index"])
            
            # Extend segment to make it longer if needed
            start_idx = max(0, segment["index"] - 1)
            end_idx = min(len(transcript) - 1, segment["index"] + 1)
            
            start_time = transcript[start_idx]["start"]
            end_time = transcript[end_idx]["end"]
            
            # Limit to reasonable duration (max 60 seconds)
            if end_time - start_time > 60:
                end_time = start_time + 60
            
            topic = extract_topic(segment["text"])
            segment_title = f"Highlight: {topic}"
            if segment["is_intro"] and not intro_selected:
                segment_title = f"Introduction: {topic}"
                intro_selected = True
            elif segment["is_ending"] and not ending_selected:
                segment_title = f"Conclusion: {topic}"
                ending_selected = True
            
            clips.append({
                "start": start_time,
                "end": end_time,
                "title": segment_title,
                "description": f"Selected based on relevance to your query"
            })
            
            remaining_slots -= 1
    
    # If we didn't find enough segments, add evenly spaced clips
    if len(clips) < num_clips:
        remaining = num_clips - len(clips)
        clip_duration = min(60, total_duration / (remaining + 1))
        
        for i in range(remaining):
            # Space clips evenly, avoiding beginning and end if already selected
            position = (i + 1.0) / (remaining + 1.0)
            
            if position < 0.2 and intro_selected:
                position = 0.3 + (i * 0.1)  # Shift away from intro
            elif position > 0.8 and ending_selected:
                position = 0.7 - (i * 0.1)  # Shift away from ending
                
            start_time = transcript[0]["start"] + (position * total_duration)
            
            # Find closest segment start
            closest_segment = min(transcript, key=lambda x: abs(x["start"] - start_time))
            start_time = closest_segment["start"]
            end_time = min(transcript[-1]["end"], start_time + clip_duration)
            
            segment_title = f"Clip {len(clips) + 1}"
            position_percent = start_time / total_duration * 100
            if position_percent < 15 and not intro_selected:
                segment_title = "Introduction"
                intro_selected = True
            elif position_percent > 85 and not ending_selected:
                segment_title = "Conclusion"
                ending_selected = True
            
            clips.append({
                "start": start_time,
                "end": end_time,
                "title": segment_title,
                "description": f"Auto-generated clip"
            })
    
    # Sort clips by start time
    clips.sort(key=lambda x: x["start"])
    
    return {
        "clips": clips,
        "narrative_structure": "Clips with introduction and conclusion"
    }

def direct_analysis_fallback(transcript, user_query, num_clips):
    """Fallback method that directly calls the LLM without using the agent framework."""
    if llm is None:
        # If LLM is not available, use heuristic approach
        return analyze_transcript_without_llm(transcript, user_query, num_clips)
        
    print("Using direct LLM analysis fallback...")
    analysis_tool = TranscriptAnalysisTool()
    return analysis_tool._run(transcript, user_query, num_clips)

# Add this function after initialize_llm
def estimate_tokens(text):
    """Estimate the number of tokens in a text string.
    This is a rough approximation - 1 token is ~4 characters for English text."""
    return len(text) // 4

# Add the following function after the collect_transcript_metrics function

def categorize_video_type(transcript):
    """Use LLM to categorize the video type based on a sample of the transcript."""
    import json
    # Use first 5 segments as a sample
    sample_transcript = transcript[:5]
    prompt = f"""You are an expert video categorization assistant. Based on the following transcript sample, please categorize the video type into one of these categories: Interview, Tutorial, Vlog, Documentary, Presentation, or Other. Provide only the category name.\nTranscript sample: {json.dumps(sample_transcript)}"""
    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            category = response.content.strip()
        else:
            category = str(response).strip()
        return category
    except Exception as e:
        print(f"Error in categorizing video type: {e}")
        return "Other"

# Modify the main function to include emotion analysis
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
    
    print(f"Selected: {selected_video}")
    
    # User Query
    default_query = "Create an engaging clip from this video"
    user_query = input(f"Enter your search query or press Enter for default: ") or default_query
    
    # Number of clips - changed default to 1
    num_clips = 1  # Default now 1 instead of 3
    try:
        num_clips_input = input(f"How many clips would you like to include? (default: {num_clips}): ")
        if num_clips_input:
            num_clips = int(num_clips_input)
            if num_clips < 1:
                print("Number of clips must be at least 1. Using default.")
                num_clips = 1
    except ValueError:
        print("Invalid input. Using default number of clips.")
    
    # New: Ask if user wants emotion-based analysis
    use_emotion = False
    emotion_type = None
    
    # Check if query is already emotion-focused
    if is_emotion_focused_query(user_query):
        print("Your query appears to be focused on emotional content.")
        use_emotion = True
        emotion_type = extract_emotion_from_query(user_query)
        if emotion_type:
            print(f"Detected emotion focus: {emotion_type}")
    
    # If emotion analysis is available, offer it as an option
    if EMOTION_ANALYSIS_AVAILABLE and not use_emotion:
        emotion_option = input("Would you like to focus on emotional content? (y/n, default: n): ").lower()
        if emotion_option == 'y':
            use_emotion = True
            emotion_types = ["happy", "sad", "angry", "fearful", "surprised", "disgust", "calm", "neutral"]
            print("Available emotion types:")
            for i, emotion in enumerate(emotion_types):
                print(f"{i+1}. {emotion}")
            
            emotion_choice = input("Select an emotion to focus on (or press Enter for any strong emotion): ")
            if emotion_choice and emotion_choice.isdigit():
                emotion_idx = int(emotion_choice) - 1
                if 0 <= emotion_idx < len(emotion_types):
                    emotion_type = emotion_types[emotion_idx]
                    print(f"Selected emotion: {emotion_type}")
    
    print(f"Using query: {user_query}")
    print(f"Number of clips: {num_clips}")
    print(f"Input video: {input_video}")
    if use_emotion:
        print(f"Using emotion analysis: {emotion_type or 'any strong emotion'}")
    
    # For emotion-only analysis, we can skip transcription
    if use_emotion and EMOTION_ANALYSIS_AVAILABLE and "transcript" not in user_query.lower():
        print("Using direct emotion analysis without transcription...")
        result = process_video_emotions(input_video, emotion_type, num_clips)
        
        if result:
            print("\nEmotional highlights found:")
            for highlight in result["highlights"]:
                print(f"{highlight['title']} ({highlight['start']:.1f}s - {highlight['end']:.1f}s)")
                print(f"  Dominant emotion: {highlight['dominant_emotion']}")
            
            print("\nOutput clips created:")
            for path in result["output_paths"]:
                print(f"  {path}")
            
            return
        else:
            print("Emotion analysis failed. Falling back to transcript-based approach.")
    
    # Step 1: Transcribe
    print("Transcribing video...")
    transcription = transcribe_video(input_video, model_name="base")
    
    if transcription is None:
        print("Transcription failed. Please check the FFmpeg installation and try again.")
        return
    
    print("Transcription completed successfully.")
    
    # Collect and display transcript metrics
    metrics = collect_transcript_metrics(transcription)
    print(f"Transcript metrics:")
    print(f" - Total duration: {metrics['duration']:.2f} seconds")
    print(f" - Segments: {metrics['segments']}")
    print(f" - Avg segment duration: {metrics['avg_segment_duration']:.2f} seconds")
    print(f" - Estimated tokens: {metrics['estimated_tokens']}")

    # Categorize video type
    video_category = "Other"
    if llm:
        print("Categorizing video type using LLM...")
        video_category = categorize_video_type(transcription)
        print(f"LLM categorized the video as: {video_category}")
        user_confirm = input("Is this categorization correct? (y/n): ").strip().lower()
        if user_confirm != 'y':
            video_category = input("Please enter the correct video category: ").strip()
    else:
        print("LLM not available for categorization. Using default category 'Other'.")
    print(f"Video category is set to: {video_category}")

    # Optimize number of clips based on transcript length
    optimal_clips = num_clips
    if metrics['estimated_tokens'] > CONFIG["token_limits"]["transcript_token_threshold"]:
        # Suggest fewer clips for very long videos
        token_ratio = metrics['estimated_tokens'] / CONFIG["token_limits"]["transcript_token_threshold"]
        if token_ratio > 2 and num_clips > 3:
            optimal_clips = max(3, num_clips // min(3, int(token_ratio)))
            if optimal_clips != num_clips:
                print(f"Note: For optimal results with this long video, consider using {optimal_clips} clips instead of {num_clips}.")
    
    # Choose the appropriate analysis approach based on user preferences
    if use_emotion and EMOTION_ANALYSIS_AVAILABLE:
        print("Using hybrid content-emotion analysis...")
        # Use the hybrid analysis tool
        hybrid_tool = HybridAnalysisTool()
        clip_data = hybrid_tool._run(transcription, input_video, user_query, num_clips)
    else:
        # Initialize the standard agent
        agent_executor = create_agent()
        
        # Prepare user query for agent
        agent_query = f"I want to create {num_clips} clips from a {video_category} video about: {user_query}. Each clip should be self-contained and engaging."
        
        # Add transcription success to chat history
        chat_history.append(HumanMessage(content="I've transcribed a video and need help creating clips from it."))
        chat_history.append(AIMessage(content="Great! I can help you analyze the transcript and create engaging video clips. What would you like the clips to be about?"))
        
        # Run the agent
        print("Analyzing transcript...")
        clip_data = None
        
        # Try the agent approach first, but only if it was created successfully
        if agent_executor:
            try:
                # Attempt to run the agent to analyze the transcript
                try:
                    agent_response = agent_executor.invoke({
                        "input": agent_query,
                        "chat_history": chat_history,
                        "transcript": transcription,
                        "num_clips": num_clips
                    })
                    
                    print("AI Analysis Complete:")
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
            print("Using direct transcript analysis...")
            analysis_tool = TranscriptAnalysisTool()
            clip_data = analysis_tool._run(transcription, user_query, num_clips)
    
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
    
    # Step 3: Create individual clips
    print(f"Creating {len(clip_data['clips'])} individual clips...")
    print(f"Narrative structure: {clip_data.get('narrative_structure', 'Not provided')}")
    
    output_paths = create_individual_clips(input_video, clip_data, output_dir)
    
    if output_paths:
        print(f"Successfully created {len(output_paths)} clips:")
        for i, path in enumerate(output_paths):
            print(f"  Clip {i+1}: {path}")
    else:
        print("Failed to create any clips.")

# Add metrics collection function
def collect_transcript_metrics(transcript):
    """Analyze transcript to collect useful metrics"""
    if not transcript:
        return {"segments": 0, "duration": 0, "avg_segment_duration": 0}
    
    total_duration = transcript[-1]["end"] - transcript[0]["start"]
    avg_segment_duration = total_duration / len(transcript)
    
    # Calculate average words per segment
    total_words = sum(len(segment["text"].split()) for segment in transcript)
    avg_words = total_words / len(transcript)
    
    # Calculate estimated tokens
    estimated_tokens = estimate_tokens(json.dumps(transcript))
    
    # Find significant content breakpoints (where there might be topic changes)
    breakpoints = []
    for i in range(1, len(transcript)):
        curr_segment = transcript[i]
        prev_segment = transcript[i-1]
        
        # Check for potential topic change (long pause between segments)
        if curr_segment["start"] - prev_segment["end"] > 3.0:  # 3 second pause
            breakpoints.append(i)
    
    metrics = {
        "segments": len(transcript),
        "duration": total_duration,
        "avg_segment_duration": avg_segment_duration,
        "avg_words_per_segment": avg_words,
        "estimated_tokens": estimated_tokens,
        "potential_breakpoints": breakpoints,
        "tokens_per_second": estimated_tokens / total_duration if total_duration > 0 else 0
    }
    
    return metrics

if __name__ == "__main__":
    main() 