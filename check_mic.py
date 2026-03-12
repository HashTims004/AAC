import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import torch

# CONFIGURATION
DEVICE_ID = 1  
DURATION = 5  
SAMPLE_RATE = 16000
OUTPUT_FILE = "debug_audio.wav"

# Add current folder to path so Whisper finds ffmpeg.exe
os.environ["PATH"] += os.pathsep + os.getcwd()

def diagnostics():
    print(" DIAGNOSTICS MODE V2")
    print("--------------------------------------------------")
    
    # 1. Record with explicit device
    device_info = sd.query_devices(DEVICE_ID, 'input')
    print(f" Using Device {DEVICE_ID}: {device_info['name']}")
    
    print(f" Recording for {DURATION} seconds... SPEAK NOW!")
    try:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                         samplerate=SAMPLE_RATE, 
                         channels=1, 
                         device=DEVICE_ID) # Force specific device
        sd.wait()
        print(" Recording complete.")
    except Exception as e:
        print(f" Recording Failed: {e}")
        return

    # 2. Volume Boost (Normalization)
    # This fixes "little to no sound" issues by maximizing the volume mathematically
    max_val = np.max(np.abs(recording))
    if max_val > 0:
        print(f" Boosting volume (Current max: {max_val:.4f})...")
        recording = recording / max_val * 0.9 # Amplify to 90% max volume
    else:
        print(" Warning: Recorded audio is pure silence (0.0). Check hardware mute switch.")

    # 3. Save File
    wav_data = (recording * 32767).astype(np.int16)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, wav_data)
    print(f" Audio saved to '{OUTPUT_FILE}'.")
    
    # 4. Transcribe
    print("--------------------------------------------------")
    print(" Attempting Transcription...")
    
    if not os.path.exists("ffmpeg.exe"):
        print(" CRITICAL ERROR: 'ffmpeg.exe' not found in project folder.")
        print(" Please download FFmpeg and place 'ffmpeg.exe' next to this script.")
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base.en", device=device)
        result = model.transcribe(OUTPUT_FILE, fp16=True)
        print(f"  AI Heard: '{result['text'].strip()}'")
    except Exception as e:
        print(f" Transcription Failed: {e}")

if __name__ == "__main__":
    diagnostics()