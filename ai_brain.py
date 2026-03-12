import torch
from transformers import pipeline

# CONFIGURATION
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ---------------------

print(f" Loading {MODEL_ID} to GPU...")

# Load model (re-using cache)
pipe = pipeline("text-generation", 
                model=MODEL_ID, 
                torch_dtype=torch.float16, 
                device_map="auto")

def generate_responses(user_input):
    """
    Forces the AI to generate 3 options by completing a pattern.
    """
    
    # "Few-Shot" Prompt
    prompt = f"""<|system|>
You are a helper for a non-verbal child. You generate 3 short, quick responses for them to choose.
</s>
<|user|>
How are you feeling?
</s>
<|assistant|>
1. I am happy.
2. I am tired.
3. I am hungry.
</s>
<|user|>
Do you want to play a game?
</s>
<|assistant|>
1. Yes, please!
2. No, thank you.
3. What game?
</s>
<|user|>
{user_input}
</s>
<|assistant|>
1."""

    # Generate
    outputs = pipe(prompt, 
                   max_new_tokens=40,   # Keep it very short
                   do_sample=True, 
                   temperature=0.6,     # Lower temperature = more focused
                   top_k=40,
                   top_p=0.90)

    generated_text = outputs[0]['generated_text']
    
    # CLEANING THE OUTPUT
    # The model will return the whole prompt + new text. We only want the new text.
    response_part = generated_text.split(f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n")[-1]
    
    # Ensure formatting
    clean_options = []
    lines = response_part.strip().split('\n')
    
    # If we force-started with "1.", we need to add it back if the model skipped it
    if lines and not lines[0].startswith("1."):
        lines[0] = "1. " + lines[0]

    for line in lines:
        # Only keep lines that look like "1. Text" or "2. Text"
        if line and (line[0].isdigit() and "." in line):
             clean_options.append(line.strip())
        
        # Stop after 3 options
        if len(clean_options) >= 3:
            break
            
    return clean_options

# TEST AREA
if __name__ == "__main__":
    test_inputs = [
        "Can you find my backpack?",
        "What do you want for lunch?"
    ]
    
    for inp in test_inputs:
        print(f"\n Input: '{inp}'")
        options = generate_responses(inp)
        print(" Options:")
        for opt in options:
            print(opt)