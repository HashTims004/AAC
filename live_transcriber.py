import sounddevice as sd
import numpy as np
import whisper
import torch
import queue
import sys

# CONFIGURATION
MODEL_TYPE = "small.en"   # Options: tiny.en, base.en, small.en, medium.en
SAMPLE_RATE = 16000       # Whisper requires 16kHz audio
BLOCK_SIZE = 30           # Block size in milliseconds
THRESHOLD = 0.02          # Silence threshold (adjust if too sensitive)
SILENCE_DURATION = 1.0    # Seconds of silence to trigger transcription
# 

print(f" Loading Whisper '{MODEL_TYPE}' model to GPU...")

# Load model to GPU (cuda)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_TYPE, device=device)

print(f" Model loaded on {device.upper()}")
print(" System is ready. Speak into your microphone...")
print("--------------------------------------------------")

q = queue.Queue()

def callback(indata, frames, time, status):
    """This function is called by the system for every chunk of audio."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def main():
    try:
        # Start recording stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            audio_buffer = []
            silence_counter = 0
            
            while True:
                # Get audio data from queue
                if not q.empty():
                    data = q.get()
                    # Calculate volume (Root Mean Square)
                    volume = np.sqrt(np.mean(data**2))
                    
                    if volume > THRESHOLD:
                        # User is speaking, add data to buffer
                        audio_buffer.append(data)
                        silence_counter = 0
                    else:
                        # Silence detected
                        if len(audio_buffer) > 0:
                            silence_counter += (len(data) / SAMPLE_RATE)
                            audio_buffer.append(data)

                        # If silence lasts long enough, process the buffer
                        if silence_counter > SILENCE_DURATION and len(audio_buffer) > 0:
                            print("Thinking...", end="\r")
                            
                            # Flatten the buffer into a single array
                            audio_np = np.concatenate(audio_buffer, axis=0).flatten()
                            
                            # Transcribe using GPU
                            result = model.transcribe(audio_np.astype(np.float32), fp16=True)
                            text = result['text'].strip()

                            if text:
                                print(f"🗣️  Heard: {text}")
                            
                            # Clear buffer for next sentence
                            audio_buffer = []
                            silence_counter = 0
                            print("🎤 Listening...  ", end="\r")

    except KeyboardInterrupt:
        print("\n Stopping...")

if __name__ == "__main__":
    main()