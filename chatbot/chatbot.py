import torch
from transformers import pipeline

class EchoSoulChatbot:
    def __init__(self, system_prompt="You are a helpful memory companion."):
        print("Loading TinyLlama-1.1B (This may take a minute)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # TinyLlama 1.1B is small enough for 8GB+ RAM
        self.pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            device_map="auto"
        )
        self.system_prompt = system_prompt
        # TinyLlama Chat template format
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_input):
        """
        Generates a response using the local LLM.
        """
        self.messages.append({"role": "user", "content": user_input})
        
        # Format prompt for TinyLlama
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        try:
            outputs = self.pipe(
                prompt, 
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.7, 
                top_k=50, 
                top_p=0.95
            )
            
            generated_text = outputs[0]["generated_text"]
            # Extract only the assistant's response (strip prompt)
            # The prompt ends with <|assistant|>
            response_start = generated_text.rfind("<|assistant|>")
            if response_start != -1:
                ai_message = generated_text[response_start + len("<|assistant|>"):].strip()
            else:
                ai_message = generated_text.strip()

            self.messages.append({"role": "assistant", "content": ai_message})
            return ai_message

        except Exception as e:
            return f"Error computing response: {str(e)}"
    
    def reset_history(self, new_system_prompt=None):
        if new_system_prompt:
            self.system_prompt = new_system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

# singleton = EchoSoulChatbot()
