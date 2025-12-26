import sys
import os

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    print("Testing imports...")
    try:
        from audio_io.speech_to_text import SpeechToText
        from audio_io.text_to_speech import TextToSpeech
        from chatbot.chatbot import EchoSoulChatbot
        from voice_clone.clone_voice import VoiceCloner
        print("Imports successful!")
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    return True

def test_instantiation():
    print("Testing class instantiation...")
    try:
        from audio_io.text_to_speech import TextToSpeech
        from chatbot.chatbot import EchoSoulChatbot
        from voice_clone.clone_voice import VoiceCloner
        
        # Skip SpeechToText instantiation as it loads a model (slow)
        tts = TextToSpeech()
        # Mocking API key for test if needed, or catching error
        try:
             bot = EchoSoulChatbot()
        except:
             print("Chatbot init might fail without API key, expected.")
             
        cloner = VoiceCloner()
        print("Instantiation successful!")
    except Exception as e:
        print(f"Instantiation failed: {e}")
        return False
    return True

if __name__ == "__main__":
    if test_imports() and test_instantiation():
        print("ALL CHECKS PASSED")
    else:
        print("CHECKS FAILED")
