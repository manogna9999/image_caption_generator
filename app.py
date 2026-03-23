import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from io import BytesIO
from gtts import gTTS
import io

# Page config FIRST
st.set_page_config(page_title="Image Caption Generator", layout="wide", initial_sidebar_state="collapsed")

# Hide ALL Streamlit UI + Perfect responsive design
st.markdown("""
<style>
    section[data-testid="stSidebar"] {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .st-emotion-cache-1r4w99b {display: none;}
    .css-1d391kg {padding-top: 1rem;}
    .css-1d391kg > div:first-child {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Perfect responsive CSS - NO SCROLLING
st.markdown("""
<style>
* {padding: 0; margin: 0;}
.main-header {font-size: 2.2rem; font-weight: 600; color: #2c3e50; 
              text-align: center; margin: 0.5rem 0 1.5rem 0; font-family: 'Segoe UI', sans-serif;}
.caption-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 1.2rem; border-radius: 12px; color: white; 
               font-size: 1.15rem; text-align: center; margin: 1rem auto; max-width: 600px;}
.btn-container {display: flex; justify-content: center; gap: 1rem; margin: 1.5rem 0;}
.stButton > button {width: 180px !important; height: 48px !important; border-radius: 10px !important; 
                    font-size: 1rem !important; font-weight: 600 !important; border: none !important;}
.preview-container {text-align: center; margin: 1rem 0; max-width: 400px; margin-left: auto; margin-right: auto;}
.preview-img {max-width: 380px !important; max-height: 280px !important; border-radius: 12px; 
              box-shadow: 0 8px 25px rgba(0,0,0,0.15); display: block; margin: 0 auto;}
.select-container {max-width: 350px; margin: 1rem auto;}
.input-container {max-width: 450px; margin: 1rem auto; text-align: center;}
.result-section {max-width: 700px; margin: 0 auto;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForVision2Seq.from_pretrained("./model")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model

if "model_ready" not in st.session_state:
    st.session_state.processor, st.session_state.model = load_model()
    st.session_state.model_ready = True

processor = st.session_state.processor
model = st.session_state.model

# Perfect centered layout
st.markdown('<h2 class="main-header">Image Caption Generator</h2>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 1rem; margin-bottom: 2rem;">' +
            'Professional image captioning system with audio narration</div>', unsafe_allow_html=True)

# Dropdown (perfectly centered)
st.markdown('<div class="select-container">', unsafe_allow_html=True)
input_method = st.selectbox("", ["Upload Image", "Camera Capture", "Image URL"])
st.markdown('</div>', unsafe_allow_html=True)

# Input section (perfectly centered)
st.markdown('<div class="input-container">', unsafe_allow_html=True)
image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose image file", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

elif input_method == "Camera Capture":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(BytesIO(camera_image.getvalue())).convert('RGB')

elif input_method == "Image URL":
    image_url = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
    if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            st.error("❌ Invalid image URL")
st.markdown('</div>', unsafe_allow_html=True)

# Preview + Buttons (perfect center alignment)
if image:
    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
    st.image(image, width=380, clamp=True, output_format="PNG")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Perfect centered buttons
    st.markdown('<div class="btn-container">', unsafe_allow_html=True)
    
    # Generate button
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("**Generate Caption**", key="generate", help="Click to generate caption"):
            device = next(model.parameters()).device
            with st.spinner("Generating caption..."):
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=50, num_beams=5)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                st.session_state.caption = caption
                st.rerun()
    
    # Listen button (only after caption generated)
    if "caption" in st.session_state:
        with col2:
            if st.button("**Listen Audio**", key="listen", help="Play caption audio"):
                tts = gTTS(text=st.session_state.caption, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.audio(audio_buffer.getvalue(), format="audio/mp3")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Caption result (perfect center)
if "caption" in st.session_state:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown(f'<div class="caption-card">**Generated Caption:**<br/>"{st.session_state.caption}"</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Perfect compact footer
st.markdown('<div style="text-align: center; margin-top: 2rem; color: #95a5a6; font-size: 0.9rem;">' +
            'Custom encoder-decoder architecture for image captioning research</div>', unsafe_allow_html=True)
