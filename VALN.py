# ==========================
# VALN Agent – Streamlit Test
# ==========================

import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import torchaudio
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime
import numpy as np
import os

# ======= CONFIG =======
BASE_DIR = Path("lifemodo_data")
MODEL_DIR = BASE_DIR / "models"
AUDIO_SNAPSHOT_DIR = BASE_DIR / "snapshots/80d26097562ad07976b92fec271572dc637e"

VISION_MODEL_PATH = MODEL_DIR / "vision_model/weights/best.pt"
LANGUAGE_MODEL_PATH = MODEL_DIR / "language_model"
AUDIO_MODEL_PATH = AUDIO_SNAPSHOT_DIR / "YamNet_float.onnx"

# Crée les dossiers temporaires si nécessaires
(BASE_DIR / "temp").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="VALN Agent Multimodal Test", layout="wide")
st.title("🧬 VALN Agent – Multimodal AI (Vision + Audio + Langage)")

# ======= LOAD MODELS =======
@st.cache_resource
def load_models():
    st.info("Chargement modèle vision...")
    vision_model = YOLO(str(VISION_MODEL_PATH))

    st.info("Chargement modèle langage...")
    tokenizer = AutoTokenizer.from_pretrained(str(LANGUAGE_MODEL_PATH))
    language_model = AutoModelForSequenceClassification.from_pretrained(str(LANGUAGE_MODEL_PATH))

    st.info("Chargement modèle audio (YamNet ONNX)...")
    audio_session = onnxruntime.InferenceSession(str(AUDIO_MODEL_PATH))

    return vision_model, audio_session, language_model, tokenizer

vision_model, audio_session, language_model, tokenizer = load_models()

# ======= VALN AGENT =======
class VALN:
    def __init__(self, vision_model, audio_session, language_model, tokenizer):
        self.vision_model = vision_model
        self.audio_session = audio_session
        self.language_model = language_model
        self.tokenizer = tokenizer

    # Vision
    def see(self, img_path):
        if not img_path or not os.path.isfile(img_path):
            return "Image non trouvée"
        img = Image.open(img_path)
        results = self.vision_model(img)
        return results

    # Audio
    def hear(self, audio_path):
        if not audio_path or not os.path.isfile(audio_path):
            return "Audio non trouvé"
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0).numpy()
        ort_inputs = {self.audio_session.get_inputs()[0].name: waveform.reshape(1, -1).astype('float32')}
        ort_outs = self.audio_session.run(None, ort_inputs)
        pred_class = int(np.argmax(ort_outs[1]))
        return pred_class

    # Langage
    def read(self, text):
        if not text or not text.strip():
            return "Aucun texte fourni"
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.language_model(**inputs)
        pred = outputs.logits.argmax().item()
        return pred

    # Fusion multimodale
    def reason(self, img_path=None, audio_path=None, text=None):
        vision_res, audio_res, text_res = None, None, None
        if img_path:
            vision_res = self.see(img_path)
        if audio_path:
            audio_res = self.hear(audio_path)
        if text:
            text_res = self.read(text)
        return vision_res, audio_res, text_res

agent = VALN(vision_model, audio_session, language_model, tokenizer)

# ======= STREAMLIT INTERFACE =======
st.sidebar.title("⚙️ Tester VALN")
mode = st.sidebar.radio("Choisir le type d'entrée :", ["📸 Image", "🎤 Audio", "📄 Texte", "🔀 Multimodal"])

# --- IMAGE ---
if mode == "📸 Image":
    uploaded_img = st.file_uploader("Upload une image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img_path = BASE_DIR / "temp/temp_img.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())
        st.image(img_path, caption="Image uploadée")
        results = agent.see(str(img_path))
        st.write("Résultats vision YOLO :", results)

# --- AUDIO ---
elif mode == "🎤 Audio":
    uploaded_audio = st.file_uploader("Upload un audio", type=["wav", "mp3"])
    if uploaded_audio:
        audio_path = BASE_DIR / "temp/temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())
        st.audio(audio_path)
        pred_class = agent.hear(str(audio_path))
        st.write("Classe audio prédite (YamNet) :", pred_class)

# --- TEXTE ---
elif mode == "📄 Texte":
    text_input = st.text_area("Entrer un texte à analyser")
    if st.button("Analyser texte"):
        pred = agent.read(text_input)
        st.write("Classe texte prédite :", pred)

# --- MULTIMODAL ---
elif mode == "🔀 Multimodal":
    uploaded_img = st.file_uploader("Upload une image", type=["png", "jpg", "jpeg"])
    uploaded_audio = st.file_uploader("Upload un audio", type=["wav", "mp3"])
    text_input = st.text_area("Entrer un texte à analyser")
    if st.button("Analyser multimodal"):
        img_path, audio_path = None, None
        if uploaded_img:
            img_path = BASE_DIR / "temp/temp_img.png"
            with open(img_path, "wb") as f:
                f.write(uploaded_img.read())
            st.image(img_path, caption="Image uploadée")
        if uploaded_audio:
            audio_path = BASE_DIR / "temp/temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.read())
            st.audio(audio_path)
        vision_res, audio_res, text_res = agent.reason(
            str(img_path) if img_path else None,
            str(audio_path) if audio_path else None,
            text_input if text_input.strip() else None
        )
        st.write("Résultats multimodaux :")
        st.write("Vision :", vision_res)
        st.write("Audio :", audio_res)
        st.write("Texte :", text_res)
