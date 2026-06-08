import os
import tempfile

# We import whisper and moviepy inside functions so that import errors 
# do not prevent the main application from starting up.
# This makes the app robust even if whisper or moviepy dependencies are missing.

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def transcribe_audio_file(audio_path, model_name="tiny"):
    """
    Transcribes a local audio file using Whisper.
    Groups segments into chunks of approx 100-150 words with timestamps.
    Returns: [{"text": str, "page_num": None, "timestamp": str}]
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "The 'openai-whisper' package is not installed. "
            "Please run: pip install openai-whisper"
        )
        
    print(f"Loading Whisper model '{model_name}' (this might take a moment on first run)...")
    try:
        # Load whisper model (uses CPU by default if GPU is not available)
        model = whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
        
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path)
    
    segments = result.get("segments", [])
    if not segments:
        text = result.get("text", "").strip()
        return [{"text": text, "page_num": None}]
        
    chunks = []
    current_chunk_text = []
    current_start = None
    
    # Group segments into ~100 word chunks or 500 characters
    char_count = 0
    
    for seg in segments:
        start_time = seg["start"]
        end_time = seg["end"]
        text = seg["text"].strip()
        
        if not text:
            continue
            
        if current_start is None:
            current_start = start_time
            
        current_chunk_text.append(text)
        char_count += len(text)
        
        # If chunk is large enough (e.g., > 600 chars), save it
        if char_count > 600 or seg == segments[-1]:
            combined_text = " ".join(current_chunk_text)
            start_str = format_timestamp(current_start)
            end_str = format_timestamp(end_time)
            
            # Use page_num as a pseudo field or append timestamp to text
            # We prefix the text with the timestamp to give LLM temporal context!
            timestamp_prefix = f"[{start_str} - {end_str}]"
            chunks.append({
                "text": f"{timestamp_prefix} {combined_text}",
                "page_num": None  # We will use this formatting in search citations
            })
            
            current_chunk_text = []
            current_start = None
            char_count = 0
            
    return chunks

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts the audio track from a video file and saves it as a WAV file.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise ImportError(
                "The 'moviepy' package is not installed. "
                "Please run: pip install moviepy"
            )
        
    print(f"Extracting audio from video: {video_path}")
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("The uploaded video file does not contain an audio track.")
            
        # Write audio track to file (parameters chosen for good quality and whisper compatibility)
        video.audio.write_audiofile(
            output_audio_path,
            codec="pcm_s16le",
            fps=16000,
            nbytes=2,
            logger=None # Disable verbose output
        )
        video.close()
    except Exception as e:
        raise RuntimeError(f"Error extracting audio from video: {str(e)}")

def process_media_file(file_path, model_name="tiny"):
    """
    Processes video or audio files, transcribes them, and returns chunks.
    Automatically detects if it's a video and extracts audio first.
    """
    ext = os.path.splitext(file_path)[1].lower()
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm']
    audio_exts = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    
    temp_audio = None
    
    try:
        if ext in video_exts:
            # Create a temporary file for audio extraction
            fd, temp_audio = tempfile.mkstemp(suffix=".wav")
            os.close(fd) # Close file descriptor, let moviepy write to path
            
            # Extract audio from video
            extract_audio_from_video(file_path, temp_audio)
            audio_to_transcribe = temp_audio
        elif ext in audio_exts:
            audio_to_transcribe = file_path
        else:
            raise ValueError(f"Unsupported media file format: {ext}")
            
        # Transcribe using Whisper
        chunks = transcribe_audio_file(audio_to_transcribe, model_name)
        return chunks
        
    finally:
        # Clean up temporary audio file if it was created
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass
