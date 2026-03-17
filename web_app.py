import streamlit as st
import streamlit.components.v1 as components
import whisper
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# PAGE CONFIGURATION
st.set_page_config(page_title="AAC AI Interface", layout="centered")

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
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)


# LOAD WHISPER (tiny.en = ~150MB)
@st.cache_resource
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("tiny.en", device=device)


# LOAD TINYLLAMA GGUF (Q4 quantized = ~670MB, runs on CPU)
@st.cache_resource
def load_tinyllama():
    model_path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    return Llama(
        model_path=model_path,
        n_ctx=512,       # Keep context small to save RAM
        n_threads=2,     # Streamlit Cloud has 2 CPU cores
        verbose=False
    )


try:
    with st.spinner("Loading hearing model (Whisper)..."):
        ears_model = load_whisper()
    with st.spinner("Loading brain model (TinyLlama)..."):
        brain_model = load_tinyllama()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# HELPER FUNCTIONS
def js_speak(text):
    components.html(f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)


def generate_responses(input_text):
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

    output = brain_model(
        prompt,
        max_tokens=60,
        temperature=0.7,
        top_k=40,
        top_p=0.90,
        stop=["</s>", "<|user|>"]  # Stop cleanly
    )

    raw = "1." + output["choices"][0]["text"]

    options = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            clean = line.split('.', 1)[-1].strip().replace('"', '').replace("'", "")
            if clean:
                options.append(clean)
        if len(options) >= 3:
            break

    while len(options) < 3:
        options.append("...")
    return options


# UI
st.title("🗣️ AAC Assistive Interface")
st.caption("Powered by Whisper + TinyLlama (Edge AI)")

if "options" not in st.session_state: st.session_state.options = []
if "last_text" not in st.session_state: st.session_state.last_text = ""

# ZONE 1: INPUT
with st.container(border=True):
    st.markdown("### 1. 🎙️ Input")
    audio_value = st.audio_input("Tap to Record")

    if audio_value:
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.read())

        with st.spinner("Transcribing..."):
            result = ears_model.transcribe("temp_input.wav", fp16=False)
            text = result['text'].strip()

        if text != st.session_state.last_text and len(text) > 1:
            st.session_state.last_text = text
            with st.spinner("Thinking..."):
                st.session_state.options = generate_responses(text)

# ZONE 2: CONTEXT
if st.session_state.last_text:
    with st.container(border=True):
        st.markdown("### 2. 🧠 Context")
        st.info(f'Heard: **"{st.session_state.last_text}"**')

# ZONE 3: RESPONSE
if st.session_state.options:
    with st.container(border=True):
        st.markdown("### 3. 💬 Response Selection")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(st.session_state.options[0], key="btn_0", use_container_width=True):
                js_speak(st.session_state.options[0])
        with col2:
            if st.button(st.session_state.options[1], key="btn_1", use_container_width=True):
                js_speak(st.session_state.options[1])
        with col3:
            if st.button(st.session_state.options[2], key="btn_2", use_container_width=True):
                js_speak(st.session_state.options[2])