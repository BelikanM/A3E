import streamlit as st
from diffusers import StableDiffusionXLPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import torch
import tempfile
import os

st.set_page_config(page_title="Fusion Drone IA - Photogrammétrie SDXL", layout="wide")

st.title("🛰️ Fusion Photogrammétrique par IA (Stable Diffusion XL)")
st.markdown("""
Ce prototype IA fusionne plusieurs images prises par drone pour créer une **carte photogrammétrique cohérente**.
Le modèle **Stable Diffusion XL + ControlNet** interprète la topographie et génère une image harmonisée.
""")

# --- Upload des images ---
uploaded_files = st.file_uploader("📸 Importez plusieurs images drone", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if len(uploaded_files) < 2:
    st.warning("Veuillez importer au moins deux images pour la fusion.")
    st.stop()

# --- Temp save images ---
temp_dir = tempfile.mkdtemp()
image_paths = []
for file in uploaded_files:
    img_path = os.path.join(temp_dir, file.name)
    with open(img_path, "wb") as f:
        f.write(file.getbuffer())
    image_paths.append(img_path)

# --- Chargement du modèle de profondeur (MiDaS) ---
@st.cache_resource
def load_depth_model():
    extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return extractor, model

extractor, depth_model = load_depth_model()

st.info("📦 Chargement du modèle Stable Diffusion XL + ControlNet...")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_sdxl_pipeline():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return pipe

pipe = load_sdxl_pipeline()

# --- Génération des cartes de profondeur ---
depth_maps = []
st.subheader("🔍 Étape 1 : Analyse de profondeur")
progress = st.progress(0)
for i, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB")
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))
    depth_maps.append(depth_img)
    progress.progress((i+1)/len(image_paths))
progress.empty()
st.success("✅ Cartes de profondeur générées")

# --- Fusion des cartes de profondeur ---
st.subheader("🧠 Étape 2 : Fusion des structures")
depth_arrays = [np.array(d).astype(np.float32) for d in depth_maps]
fused_depth = np.mean(depth_arrays, axis=0)
fused_depth_img = Image.fromarray(fused_depth.astype(np.uint8))
st.image(fused_depth_img, caption="Carte de profondeur fusionnée", use_container_width=True)

# --- Fusion par SDXL ---
st.subheader("🎨 Étape 3 : Génération de la carte photogrammétrique")
prompt = st.text_area("🗣️ Décrivez la scène (facultatif)", 
                      "Aerial orthomosaic map reconstructed from multiple drone images, realistic colors, consistent topography, high resolution")

if st.button("🚀 Fusionner les images"):
    with st.spinner("Fusion IA en cours... cela peut prendre 1 à 3 minutes selon votre GPU"):
        result = pipe(prompt, image=fused_depth_img, num_inference_steps=40).images[0]
        st.image(result, caption="🛰️ Résultat photogrammétrique fusionné", use_container_width=True)

        # Sauvegarde du résultat
        result_path = os.path.join(temp_dir, "fusion_result.png")
        result.save(result_path)
        with open(result_path, "rb") as f:
            st.download_button("📥 Télécharger l'image fusionnée", f, file_name="fusion_result.png")

st.markdown("---")
st.caption("© 2025 - IA Photogrammétrie SDXL | Propulsé par Stable Diffusion XL + ControlNet + MiDaS")
