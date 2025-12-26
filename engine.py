import os
import json
import shutil
import torch
import torchaudio

# Monkeypatch torchaudio for SpeechBrain compatibility (still helpful to keep here or move to patch_utils, 
# but patch_utils is better. For now seeing as I didn't move it yet, I will keep torchaudio patch here for safety or relying on patch_utils if I moved it? 
# Wait, I didn't verify if I moved torchaudio patch to patch_utils. 
# Checking patch_utils content... I ONLY moved symlinks. 
# So I must KEEP torchaudio patch here.)

# Monkeypatch torchaudio for SpeechBrain compatibility (Fixes AttributeError: 'module' object has no attribute 'list_audio_backends')
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import static_ffmpeg
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
import whisper
import soundfile as sf
import numpy as np

# Ensure ffmpeg is available
static_ffmpeg.add_paths()

class EchoSoulEngine:
    def __init__(self):
        print("Initializing EchoSoul Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Ensure personas directory exists
        os.makedirs("personas", exist_ok=True)

        # 1. Load Chatbot (TinyLlama)
        print("Loading Chat Model (TinyLlama)...")
        self.chat_pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            device_map="auto"
        )
        self.chat_history = []
        self.current_persona_name = "default"
        self.system_prompt = "You are a helpful AI assistant." # Default fallback
        
        # 2. Load STT (Whisper)
        print("Loading Hearing Model (Whisper)...")
        self.stt_model = whisper.load_model("base", device=self.device)

        # 3. Load TTS (SpeechT5)
        print("Loading Voice Model (SpeechT5)...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # 4. Load Voice Encoder (SpeechBrain) for Cloning
        print("Loading Voice Encoder (SpeechBrain)...")
        self.voice_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="tmp_model",
            run_opts={"device": self.device}
        )
        
        # Load default speaker embedding
        self._load_default_speaker_embeddings()
        
        print("EchoSoul Engine Ready!")

    def _load_default_speaker_embeddings(self):
        try:
            # CMS Arctic dataset can be flakey, using a local fallback tensor if it fails
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Note: Default voice dataset skipped ({e}). Initializing with neutral voice.")
            # Standard neutral/random embedding (1, 512)
            self.speaker_embeddings = torch.zeros(1, 512).to(self.device)
            # Add some slight noise to make it not completely robotic
            self.speaker_embeddings.normal_(mean=0, std=0.01)

    def clone_voice(self, audio_path):
        """Generates an x-vector embedding from an audio file."""
        if not audio_path: return "No audio provided."
        try:
            # Manual load using Librosa to support M4A/MP3 (via ffmpeg)
            # automatically resamples to 16000Hz which is what we need
            import librosa
            
            # Force 'audioread' backend which uses ffmpeg for m4a support
            # soundfile (default) often fails with m4a on windows
            # We explicitly tell librosa to try audioread if available
            try:
                signal, fs = librosa.load(audio_path, sr=16000)
            except Exception:
                 # If default fails, force audioread
                 # Note: librosa 0.10 deprecated 'res_type', but backend selection is implicit. 
                 # We can try to force it by importing audioread directly if needed, 
                 # but usually librosa falls back. If it didn't, we might need to be more aggressive.
                 # Let's try explicitly using audioread via librosa if we can, or just catch the error and try a different method.
                 # Actually, the error 'Format not recognised' comes from soundfile. 
                 # Unsetting the default backend might help if librosa > 0.10
                 import os
                 os.environ['LIBROSA_AUDIO_BACKEND'] = 'audioread'
                 signal, fs = librosa.load(audio_path, sr=16000)
            
            # 2. Convert to Tensor
            signal = torch.from_numpy(signal).float()
            
            # 3. Prepare for Encoder (Batch, Time)
            signal = signal.unsqueeze(0).to(self.device)
            
            # 4. Encode
            embeddings = self.voice_encoder.encode_batch(signal)
            
            # Resize from (1, 1, 512) to (1, 512)
            self.speaker_embeddings = embeddings.squeeze(1).to(self.device)
            
            return "Voice cloned successfully!"
        except Exception as e:
            print(f"Cloning Error: {e}")
            return f"Failed to clone voice: {e}"

    def configure_persona(self, name, relationship, personality, memories):
        """Sets the system prompt from structured inputs and loads previous context if available."""
        self.current_persona_name = name
        
        # Try to load existing memory
        loaded = self.load_context(name)
        if loaded:
            return f"Welcome back. Loaded existing memory for {name}."
            
        # Construct the System Prompt
        self.system_prompt = (
            f"You are {name}. You are the user's {relationship}. "
            f"Your personality is {personality}. "
            f"You cherish these memories: {memories}. "
            "You are talking to your loved one. Keep responses short, warm, and highly conversational. "
            "Never act like an AI. You are a soul preserved in memory."
        )
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        return f"Persona created: {name} ({relationship})"

    def save_context(self):
        """Saves the current persona and chat history to disk."""
        data = {
            "system_prompt": self.system_prompt,
            "chat_history": self.chat_history
        }
        # Save JSON
        with open(f"personas/{self.current_persona_name}.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # Save Voice Embedding
        if hasattr(self, 'speaker_embeddings'):
            torch.save(self.speaker_embeddings, f"personas/{self.current_persona_name}.pt")
        print(f"Saved memory for {self.current_persona_name}")

    def load_context(self, name):
        """Loads persona and voice if they exist."""
        json_path = f"personas/{name}.json"
        pt_path = f"personas/{name}.pt"
        
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                self.system_prompt = data["system_prompt"]
                self.chat_history = data["chat_history"]
            
            if os.path.exists(pt_path):
                self.speaker_embeddings = torch.load(pt_path, map_location=self.device)
            
            return True
        return False

    def transcribe(self, audio_path):
        if not audio_path: return ""
        try:
            result = self.stt_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    def chat(self, user_text):
        if not user_text: return ""
        
        self.chat_history.append({"role": "user", "content": user_text})
        
        # Format prompt
        prompt = self.chat_pipe.tokenizer.apply_chat_template(
            self.chat_history, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.chat_pipe(
            prompt, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.6,
            repetition_penalty=1.2,
            top_k=50
        )
        
        generated = outputs[0]["generated_text"]
        token = "<|assistant|>"
        if token in generated:
            ai_msg = generated.split(token)[-1].strip()
        else:
            ai_msg = generated.strip()
            
        self.chat_history.append({"role": "assistant", "content": ai_msg})
        
        # Auto-save after every turn
        self.save_context()
        
        return ai_msg

    def speak(self, text, output_file="response.wav"):
        if not text: return None
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            speech = self.tts_model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
            return output_file
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def process_turn(self, audio_input=None, text_input=None):
        """
        Main entry point. Handles Audio -> STT -> Chat -> TTS -> Audio/Text
        """
        user_msg = ""
        
        # 1. INPUT: Audio takes precedence
        if audio_input:
            user_msg = self.transcribe(audio_input)
        elif text_input:
            user_msg = text_input
            
        if not user_msg:
            return [], None # Empty turn

        # 2. THINK: Chatbot
        ai_response = self.chat(user_msg)

        # 3. SPEAK: Generate Audio
        ai_audio = self.speak(ai_response)
        
        # Return chat log format (list of dicts for Gradio Chatbot type="messages")
        # We skip the system prompt (index 0) to avoid showing internal instructions
        return self.chat_history[1:], ai_audio
