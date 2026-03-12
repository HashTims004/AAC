import tkinter as tk
from tkinter import ttk, font
import threading
import sounddevice as sd
import numpy as np
import whisper
import torch
import pyttsx3
from transformers import pipeline
import queue
import time

# CONFIGURATION 
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.2
DEVICE_ID = 1  # Mic ID
THEME_BG = "#000000"       
THEME_ACCENT = "#00ff41"   
THEME_TEXT = "#ffffff"

class AACSmartwatch:
    def __init__(self, root):
        self.root = root
        self.root.title("AAC Wearable")
        self.root.geometry("350x450")
        self.root.configure(bg=THEME_BG)
        self.root.resizable(False, False)

        # UI LAYOUT
        # 1. Status Bar (Top)
        self.status_label = tk.Label(root, text="INITIALIZING...", font=("Arial", 10, "bold"), fg="yellow", bg=THEME_BG)
        self.status_label.pack(pady=10)

        # 2. Transcription Box (What the watch hears)
        self.hearing_label = tk.Label(root, text="Listening...", font=("Arial", 14, "italic"), fg="#888888", bg=THEME_BG, wraplength=300)
        self.hearing_label.pack(pady=20)

        # 3. Response Buttons (The "Minimal Motor Input")
        self.buttons = []
        for i in range(3):
            btn = tk.Button(root, text=f"Option {i+1}", font=("Arial", 12), 
                          bg="#222222", fg="white", activebackground=THEME_ACCENT,
                          width=30, height=3, state="disabled",
                          command=lambda idx=i: self.speak_selection(idx))
            btn.pack(pady=5)
            self.buttons.append(btn)

        # AI INITIALIZATION (Background)
        self.q = queue.Queue()
        self.options_text = [] # Store current text options
        
        # Start the heavy loading in a separate thread to keep UI responsive
        threading.Thread(target=self.load_ai_models, daemon=True).start()

    def load_ai_models(self):
        """Loads Whisper and TinyLlama in background"""
        try:
            # Voice
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)

            # Ears
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ears_model = whisper.load_model("base.en", device=device)

            # Brain
            self.brain_pipe = pipeline("text-generation", 
                                     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                                     torch_dtype=torch.float16, 
                                     device_map="auto")
            
            # Start Listening Loop
            self.update_status(" LISTENING...", THEME_ACCENT)
            self.start_listening()
            
        except Exception as e:
            self.update_status(f"ERROR: {str(e)}", "red")

    def update_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def update_hearing(self, text):
        self.hearing_label.config(text=f'"{text}"', fg="white")

    def update_buttons(self, options):
        self.options_text = options
        for i, btn in enumerate(self.buttons):
            if i < len(options):
                btn.config(text=options[i], state="normal", bg="#333333")
            else:
                btn.config(text="", state="disabled", bg=THEME_BG)

    def speak_selection(self, idx):
        """User tapped a button -> Speak it"""
        text_to_speak = self.options_text[idx]
        self.update_status(f" SPEAKING...", "cyan")
        
        # Disable buttons while speaking
        for btn in self.buttons: btn.config(state="disabled")
        
        # Speak in a separate thread so UI doesn't freeze
        threading.Thread(target=self._speak_thread, args=(text_to_speak,)).start()

    def _speak_thread(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        time.sleep(1)
        # Reset UI
        self.update_status(" LISTENING...", THEME_ACCENT)
        self.hearing_label.config(text="...", fg="#888888")
        for btn in self.buttons: 
            btn.config(text="", state="disabled", bg="#222222")

    # AUDIO LOGIC
    def audio_callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def start_listening(self):
        """The main loop that runs forever in the background"""
        def listen_loop():
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self.audio_callback, device=DEVICE_ID):
                audio_buffer = []
                silence_counter = 0

                while True:
                    if not self.q.empty():
                        data = self.q.get()
                        volume = np.sqrt(np.mean(data**2))

                        if volume > SILENCE_THRESHOLD:
                            audio_buffer.append(data)
                            silence_counter = 0
                            # Optional: Visual indicator of noise could go here
                        else:
                            if len(audio_buffer) > 0:
                                silence_counter += (len(data) / SAMPLE_RATE)
                                audio_buffer.append(data)

                            # TRIGGER PROCESSING
                            if silence_counter > SILENCE_DURATION and len(audio_buffer) > 0:
                                self.update_status(" THINKING...", "orange")
                                
                                # Process Audio
                                audio_np = np.concatenate(audio_buffer, axis=0).flatten()
                                max_val = np.max(np.abs(audio_np))
                                if max_val > 0: audio_np = audio_np / max_val * 0.9

                                result = self.ears_model.transcribe(audio_np.astype(np.float32), fp16=True)
                                input_text = result['text'].strip()

                                if len(input_text) > 2:
                                    self.update_hearing(input_text)
                                    
                                    # Generate Response
                                    options = self.generate_smart_responses(input_text)
                                    self.update_buttons(options)
                                    self.update_status(" SELECT OPTION", "white")
                                else:
                                    # False alarm (noise), reset
                                    self.update_status(" LISTENING...", THEME_ACCENT)

                                audio_buffer = []
                                silence_counter = 0
                            
                            # Safety clear
                            if len(audio_buffer) > 300: audio_buffer = []

        threading.Thread(target=listen_loop, daemon=True).start()

    def generate_smart_responses(self, input_text):
        prompt = f"""<|system|>
You are a communication assistant for a person with CP.
Suggest 3 short, first-person responses.
</s>
<|user|>
Where are you going?
</s>
<|assistant|>
1. I am going home.
2. To the park.
3. I don't know yet.
</s>
<|user|>
{input_text}
</s>
<|assistant|>
1."""
        # Increased max_new_tokens to 60 to prevent cut-off sentences
        outputs = self.brain_pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.7, top_k=50)
        generated_text = outputs[0]['generated_text']
        response_part = generated_text.split(f"<|user|>\n{input_text}\n</s>\n<|assistant|>\n")[-1]
        
        options = []
        lines = response_part.strip().split('\n')
        if lines and not lines[0].startswith("1."): lines[0] = "1. " + lines[0]

        for line in lines:
            if line and (line[0].isdigit() and "." in line):
                 clean_text = line.split('.', 1)[-1].strip()
                 # Safety: remove any generated quotes
                 clean_text = clean_text.replace('"', '').replace("'", "")
                 options.append(clean_text)
            if len(options) >= 3: break
        
        # Fallback if AI fails to generate 3 options
        while len(options) < 3:
            options.append("...")
            
        return options

# LAUNCHER 
if __name__ == "__main__":
    root = tk.Tk()
    app = AACSmartwatch(root)
    root.mainloop()