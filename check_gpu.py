import torch

def check_system():
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("Success: CUDA is available!")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Error: CUDA not detected. The AI will run slowly on CPU.")

if __name__ == "__main__":
    check_system()