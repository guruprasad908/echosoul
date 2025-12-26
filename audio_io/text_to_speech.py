import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import os
import numpy as np

class TextToSpeech:
    def __init__(self):
        print("Loading SpeechT5 for TTS...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Load xvector embeddings for a default speaker
        print("Loading speaker embeddings...")
        try:
            # Try to load with force_redownload to fix potential cache corruption
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True, download_mode="force_redownload")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Warning: Failed to load speaker embeddings ({e}). Using random voice.")
            # Fallback: Random speaker embedding (dim 512 for SpeechT5)
            self.speaker_embeddings = torch.randn(1, 512).to(self.device)

    def synthesize(self, text, output_file="output.wav"):
        """
        Synthesizes text to speech using SpeechT5.
        """
        if not text:
            return None
        
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Generate speech
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            
            # Save to file
            sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
            return output_file
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

# Singleton
# tts = TextToSpeech()
