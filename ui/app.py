import gradio as gr
import sys
import os

# 1. Setup path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Apply Patches for Windows Permissions (Must be before engine import)
import patch_utils
patch_utils.apply_patches()

# 3. Import Dependencies
from engine import EchoSoulEngine

# --- Initialization ---
engine = EchoSoulEngine()
theme = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")

# --- Logic Handlers ---
def step1_submit(name, relation, personality, memories):
    if not name: return "Please enter a name.", gr.Tabs(selected=0)
    msg = engine.configure_persona(name, relation, personality, memories)
    return msg, gr.Tabs(selected=1) # Move to Step 2

def step2_submit(audio_path):
    if not audio_path: return "No audio uploaded.", None, gr.Tabs(selected=1)
    msg = engine.clone_voice(audio_path)
    # Generate a greeting sample
    sample = engine.speak("I am ready to talk to you.")
    return f"{msg} Listen to the sample below.", sample, gr.Tabs(selected=1)

def enter_sanctuary():
    return gr.Tabs(selected=2) # Move to Step 3

def chat_interaction(audio, text, history):
    chat_log, ai_audio = engine.process_turn(audio_input=audio, text_input=text)
    return chat_log, ai_audio


# --- UI Construction ---
with gr.Blocks(title="EchoSoul AI", theme=theme, css="footer {visibility: hidden}") as demo:
    
    # Header
    gr.Markdown(
        """
        # ✨ EchoSoul AI | Memory Companion
        *Preserve the memory. Keep the conversation alive.*
        """
    )

    with gr.Tabs() as tabs:
        
        # --- TAB 1: PROFILE CREATION ---
        with gr.TabItem("1. Create Profile", id=0):
            with gr.Column(variant="panel"):
                gr.Markdown("### 📝 Who are we remembering?")
                
                with gr.Row():
                    name_input = gr.Textbox(label="Name", placeholder="e.g. Grandma Rose")
                    relation_input = gr.Dropdown(
                        ["Parent", "Grandparent", "Partner", "Child", "Friend", "Mentor", "Pet"], 
                        label="Relationship", value="Grandparent"
                    )
                
                personality_input = gr.Dropdown(
                    ["Warm & Kind", "Stern & Wise", "Funny & Energetic", "Calm & Listener", "Professional"],
                    label="Personality Type", value="Warm & Kind"
                )
                
                memories_input = gr.Textbox(
                    label="Key Memories (The more details, the better)", 
                    lines=4, 
                    placeholder="e.g. She loved gardening, especially roses. She always told stories about the 1960s. She called me 'Pumpkin'."
                )
                
                step1_btn = gr.Button("Next Step: Voice Lab ➡️", variant="primary")
                step1_status = gr.Markdown()

        # --- TAB 2: VOICE LAB ---
        with gr.TabItem("2. Voice Lab", id=1):
             with gr.Column(variant="panel"):
                gr.Markdown("### 🎙️ Capture the Soul's Voice")
                gr.Markdown("Upload a clear 10-second recording of their voice. If you don't have one, we will use a gentle generic voice.")
                
                voice_upload = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio Clip")
                
                with gr.Row():
                    process_voice_btn = gr.Button("🧪 Analyze Voice", variant="secondary")
                    sample_audio_out = gr.Audio(label="Greeting Sample", interactive=False)
                
                voice_status = gr.Markdown()
                
                step2_btn = gr.Button("Enter Sanctuary 🕊️", variant="primary")

        # --- TAB 3: THE SANCTUARY ---
        with gr.TabItem("3. The Sanctuary", id=2):
             with gr.Column():
                chatbot_display = gr.Chatbot(
                    label="Conversation", 
                    height=550, 
                    type="messages",
                    avatar_images=(None, "https://api.dicebear.com/9.x/micah/svg?seed=Grandma") # Generic Avatar
                )
                
                with gr.Row():
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
                    msg_input = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, scale=4)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                audio_output = gr.Audio(label="EchoSoul's Voice", interactive=False, autoplay=True, visible=True) # Always visible now

    # --- Interactions ---
    
    # Step 1 -> Step 2
    step1_btn.click(
        step1_submit, 
        inputs=[name_input, relation_input, personality_input, memories_input], 
        outputs=[step1_status, tabs]
    )
    
    # Step 2 Logic
    process_voice_btn.click(
        step2_submit,
        inputs=[voice_upload],
        outputs=[voice_status, sample_audio_out, tabs] # Can stay on tab 1 if error, logic handles it
    )
    
    step2_btn.click(enter_sanctuary, outputs=[tabs])
    
    # Chat Logic
    msg_input.submit(chat_interaction, inputs=[mic_input, msg_input, chatbot_display], outputs=[chatbot_display, audio_output])
    send_btn.click(chat_interaction, inputs=[mic_input, msg_input, chatbot_display], outputs=[chatbot_display, audio_output])
    mic_input.change(chat_interaction, inputs=[mic_input, msg_input, chatbot_display], outputs=[chatbot_display, audio_output]) # Auto-send on audio stop

if __name__ == "__main__":
    demo.launch()
