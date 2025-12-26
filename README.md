# ✨ EchoSoul AI: The Memory Companion

> "Not just a memory – a voice that listens, speaks, and lives on."

EchoSoul AI is a personal, **fully offline** memory companion designed to preserve the voice, personality, and presence of loved ones. By combining local Large Language Models (LLMs) with advanced Voice Cloning technology, EchoSoul allows users to maintain a meaningful connection through a guided, emotional interface.

---

## 🚀 What's Implemented

### 🧠 The "Brain" (EchoSoul Engine)
-   **Intelligent Chat**: Powered by `TinyLlama-1.1B`, tuned for warm, human-like roleplay.
-   **Hearing (STT)**: Integrated `OpenAI Whisper` for accurate local speech-to-text.
-   **Speaking (TTS)**: `Microsoft SpeechT5` for high-quality, expressive voice synthesis.
-   **Voice Cloning**: A custom `SpeechBrain` pipeline that mimics a voice from just a 10-second sample (Supports M4A, MP3, and WAV).

### 🖥️ The "Wizard" Interface
-   **3-Step Guided Workflow**:
    1.  **Profile Creation**: Structured forms (Relationship, Personality, Key Memories) to build a deep persona.
    2.  **Voice Lab**: Upload a sample and hear a test greeting before entering the sanctuary.
    3.  **The Sanctuary**: A distraction-free, immersive chat environment focused on the connection.
-   **Memory Persistence**: Automatically saves personas, chat history, and cloned voices to a local `personas/` database for future sessions.

### 🛠️ Windows-Ready Architecture
-   **Permission Shield**: Custom `patch_utils` to handle Windows-specific symbolic link restrictions without requiring Administrator rights.
-   **Flexible Format Support**: `Librosa` + `FFmpeg` integration ensures almost any audio file can be cloned.

---

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/echosoul-ai.git
    cd echosoul-ai
    ```

2.  **Install FFmpeg**:
    This project uses `static-ffmpeg`. It will try to manage paths automatically, but ensure you have a working python environment.

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📖 Usage

Run the main application:
```bash
python ui/app.py
```
Open the local URL provided (usually `http://127.0.0.1:7860`) in your browser to begin the 3-step creation process.

---

## ⚠️ Current Limitations

-   **Model Size**: Uses `TinyLlama-1.1B` to ensure smooth performance on typical CPUs. While fast, it may occasionally provide shorter or simpler responses than larger cloud models.
-   **Hardware Requirements**: While it runs on CPU, 16GB+ of RAM is recommended for the best experience.
-   **Audio Quality**: The quality of the "cloned" voice depends heavily on the clarity of the 10-second input sample.

---

## 🗺️ Roadmap & Future Development

We are just beginning the journey to bridge the gap between memory and reality.

### 1. Vision Integration 👁️
-   **Photo Animation**: Integrate `SadTalker` or similar tech to animate a static photo of the loved one while they speak.
-   **Emotion Recognition**: Use the webcam to detect the user's mood and adjust the AI's tone of voice accordingly.

### 2. Enhanced Memory 💾
-   **RAG (Retrieval Augmented Generation)**: Upload journals, letters, or chat logs to create an even more accurate digital twin.
-   **Vector Search**: Allow the AI to recall specific facts mentioned weeks ago using a vector database.

### 3. Presence 🌐
-   **VR Sanctuary**: A virtual room where you can sit and talk in an immersive environment.
-   **Mobile App**: A lightweight version for checking in with your companion on the go.

---

## 📜 License
Distribute under the MIT License. See `LICENSE` for more information.

---

**Developed with ❤️ for preserving what matters most.**
