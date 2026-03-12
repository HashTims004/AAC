import sounddevice as sd
import numpy as np
import whisper
import torch
import pyttsx3
from transformers import pipeline
import queue
import sys
import os

# CONFIGURATION 
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02    # Adjust if environment is noisy
SILENCE_DURATION = 1.2      # Seconds of silence to trigger processing
DEVICE_ID = 1               # MIC ID 

# 1. SETUP TEXT-TO-SPEECH (The Voice)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# 2. SETUP WHISPER (The Ears)
print(" Loading Hearing (Whisper)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
ears_model = whisper.load_model("base.en", device=device)

# 3. SETUP TINYLLAMA (The Brain)
print(" Loading Intelligence (TinyLlama)...")
brain_pipe = pipeline("text-generation", 
                      model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                      torch_dtype=torch.float16, 
                      device_map="auto")

# 4. AUDIO QUEUE
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status: print(status, file=sys.stderr)
    q.put(indata.copy())

def speak_out_loud(text):
    """Speaks the text using system voice"""
    print(f" Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def get_smart_responses(input_text):
    """Generates 3 first-person responses"""
    # IMPROVED PROMPT: Forces First-Person Perspective
    prompt = f"""<|system|>
You are a communication assistant for a person with CP.
Read the incoming text and suggest 3 short, casual, first-person responses for the user to say back.
Avoid robot-like answers.
</s>
<|user|>
Where is your bag?
</s>
<|assistant|>
1. It is in my room.
2. I left it at school.
3. I don't know.
</s>
<|user|>
{input_text}
</s>
<|assistant|>
1."""

    outputs = brain_pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.7, top_k=50)
    generated_text = outputs[0]['generated_text']
    
    # Extract just the new options
    response_part = generated_text.split(f"<|user|>\n{input_text}\n</s>\n<|assistant|>\n")[-1]
    
    # Clean and format
    options = []
    lines = response_part.strip().split('\n')
    if lines and not lines[0].startswith("1."): lines[0] = "1. " + lines[0] # Fix first line if needed

    for line in lines:
        if line and (line[0].isdigit() and "." in line):
             # Remove the number (e.g., "1. Yes" -> "Yes") for cleaner TTS later
             clean_text = line.split('.', 1)[-1].strip()
             options.append(clean_text)
        if len(options) >= 3: break
            
    return options

def main():
    print("\n SYSTEM READY.")
    print(f" Listening on Device {DEVICE_ID}...")
    print("---------------------------------------")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, device=DEVICE_ID):
            audio_buffer = []
            silence_counter = 0
            is_listening = True

            while True:
                if not q.empty():
                    data = q.get()
                    volume = np.sqrt(np.mean(data**2))

                    # Logic: If loud, keep recording. If silent, count up.
                    if volume > SILENCE_THRESHOLD:
                        audio_buffer.append(data)
                        silence_counter = 0
                        print(" Listening...   ", end="\r")
                    else:
                        if len(audio_buffer) > 0:
                            silence_counter += (len(data) / SAMPLE_RATE)
                            audio_buffer.append(data)

                        # TRIGGER PROCESSING
                        if silence_counter > SILENCE_DURATION and len(audio_buffer) > 0:
                            print("\n Processing Audio...", end="\r")
                            
                            # 1. Transcribe
                            audio_np = np.concatenate(audio_buffer, axis=0).flatten()
                            
                            # Boost volume (Normalization) to fix 'quiet mic' issues
                            max_val = np.max(np.abs(audio_np))
                            if max_val > 0: audio_np = audio_np / max_val * 0.9

                            result = ears_model.transcribe(audio_np.astype(np.float32), fp16=True)
                            input_text = result['text'].strip()

                            if len(input_text) > 2: # Ignore empty noise
                                print(f"\n HEARD: '{input_text}'")
                                
                                # 2. Generate Options
                                print(" Thinking...")
                                options = get_smart_responses(input_text)
                                
                                # 3. Display Interface
                                print("\n SELECT A RESPONSE (Type 1, 2, or 3):")
                                for i, opt in enumerate(options):
                                    print(f"[{i+1}] {opt}")
                                
                                # 4. Simulate User Selection (In real life, this is a tap)
                                # For this prototype, we pause and wait for keyboard input
                                try:
                                    choice = input(" Your Choice > ")
                                    if choice in ['1', '2', '3'] and int(choice) <= len(options):
                                        selected_text = options[int(choice)-1]
                                        speak_out_loud(selected_text)
                                    else:
                                        print(" Invalid selection or skipped.")
                                except ValueError:
                                    pass

                            # Reset for next turn
                            audio_buffer = []
                            silence_counter = 0
                            print("\n---------------------------------------")
                            print(" Listening...")
                        
                        # If buffer gets too long without silence (background noise), clear it
                        if len(audio_buffer) > 300: # ~10 seconds
                             audio_buffer = []

    except KeyboardInterrupt:
        print("\n System Offline.")

if __name__ == "__main__":
    main()