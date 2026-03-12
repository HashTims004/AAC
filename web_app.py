import streamlit as st
import streamlit.components.v1 as components
import whisper
import torch
from transformers import pipeline

# PAGE CONFIGURATION
st.set_page_config(page_title="AAC AI Interface", layout="centered")

# Custom CSS for better segmentation
st.markdown("""
    <style>
    .stButton>button {
        height: 80px;
        font-size: 20px !important;
        background-color: #f0f2f6;
        border-radius: 10px;
        border: 2px solid #d1d5db;
    }
    .stButton>button:hover {
        border-color: #00ff41;
        background-color: #e6ffec;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# LOAD AI MODELS (Cached)
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ears_model = whisper.load_model("base.en", device=device)
    brain_pipe = pipeline("text-generation", 
                          model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                          torch_dtype=torch.float16, 
                          device_map="auto")
    return ears_model, brain_pipe

try:
    ears_model, brain_pipe = load_models()
except Exception as e:
    st.error(f" Error loading models: {e}")
    st.stop()

# HELPER FUNCTIONS 
def js_speak(text):
    """Client-side Text-to-Speech"""
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

def generate_responses(input_text):
    """Generates 3 contextual options"""
    prompt = f"""<|system|>
You are a helper for a non-verbal child. Suggest 3 short, first-person responses.
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
    
    outputs = brain_pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)
    generated_text = outputs[0]['generated_text']
    response_part = generated_text.split(f"<|user|>\n{input_text}\n</s>\n<|assistant|>\n")[-1]
    
    options = []
    lines = response_part.strip().split('\n')
    if lines and not lines[0].startswith("1."): lines[0] = "1. " + lines[0]

    for line in lines:
        if line and (line[0].isdigit() and "." in line):
            clean_text = line.split('.', 1)[-1].strip().replace('"', '').replace("'", "")
            options.append(clean_text)
        if len(options) >= 3: break
    
    while len(options) < 3: options.append("...")
    return options

# UI SEGMENTATION

st.title(" AAC Assistive Interface")
st.caption("Powered by Edge AI (Whisper + TinyLlama)")

# Initialize State
if "options" not in st.session_state: st.session_state.options = []
if "last_text" not in st.session_state: st.session_state.last_text = ""

# ZONE 1: INPUT (The Ears)
with st.container(border=True):
    st.markdown("### 1.  Input")
    audio_value = st.audio_input("Tap to Record")

    if audio_value:
        # Save temp file
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.read())
        
        # Transcribe
        result = ears_model.transcribe("temp_input.wav", fp16=True)
        text = result['text'].strip()
        
        # Update State only if new
        if text != st.session_state.last_text and len(text) > 1:
            st.session_state.last_text = text
            st.session_state.options = generate_responses(text)

# ZONE 2: CONTEXT (The Brain)
if st.session_state.last_text:
    with st.container(border=True):
        st.markdown("### 2.  Context")
        st.info(f"Heard: **\"{st.session_state.last_text}\"**")

# ZONE 3: OUTPUT (The Voice)
if st.session_state.options:
    with st.container(border=True):
        st.markdown("### 3.  Response Selection")
        
        col1, col2, col3 = st.columns(3)
        
        # Option 1
        with col1:
            if st.button(st.session_state.options[0], key="btn_0", use_container_width=True):
                js_speak(st.session_state.options[0])
        
        # Option 2
        with col2:
            if st.button(st.session_state.options[1], key="btn_1", use_container_width=True):
                js_speak(st.session_state.options[1])
        
        # Option 3
        with col3:
            if st.button(st.session_state.options[2], key="btn_2", use_container_width=True):
                js_speak(st.session_state.options[2])