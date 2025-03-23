from gtts import gTTS
import os
import uuid

def convert_to_hindi_speech(text: str) -> str:
    """
    Convert text to Hindi speech using Google TTS.
    
    Args:
        text: Text to convert to speech (should be in Hindi)
        
    Returns:
        Path to the generated audio file
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = "audio_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(output_dir, filename)
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Error converting text to Hindi speech: {e}")
        return ""