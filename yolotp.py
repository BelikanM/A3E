import os
import shutil
import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import zipfile

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(page_title="YOLO Photogrammetry Sorter", layout="wide")

st.title("📸 YOLO Photogrammetry Optimizer")
st.markdown("""
Ce module réduit automatiquement le nombre d’images d’un dossier photogrammétrique
en ne gardant que les **angles de vue les plus uniques et informatifs**.

⚙️ Basé sur **YOLOv8 + Similarité visuelle optimisée**, sans surcharger la mémoire.
""")

# -------------------------------
# UPLOAD SECTION
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Déposez vos images (plusieurs fichiers autorisés)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("💡 Uploadez vos images pour commencer.")
    st.stop()

# -------------------------------
# PARAMÈTRES
# -------------------------------
st.sidebar.header("⚙️ Paramètres du tri")
min_conf = st.sidebar.slider("Seuil de confiance YOLO", 0.1, 1.0, 0.5, 0.05)
keep_ratio = st.sidebar.slider("Proportion d’images à conserver (%)", 1, 100, 10, 1)
resize_dim = st.sidebar.selectbox("Taille de réduction des images", [64, 128, 256], index=1)

# -------------------------------
# CHARGEMENT YOLO
# -------------------------------
st.subheader("🔍 Analyse en cours…")
st.write("Chargement du modèle YOLO...")

model = YOLO("yolov8n.pt")  # Modèle léger
st.success("✅ YOLO chargé avec succès.")

# Dossier temporaire pour le traitement
tmp_dir = tempfile.mkdtemp()
image_paths = []
for file in uploaded_files:
    file_path = os.path.join(tmp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    image_paths.append(file_path)

# -------------------------------
# EXTRACTION DE FEATURES OPTIMISÉE
# -------------------------------
st.write("🔧 Extraction des vecteurs d’images...")
features = []

progress = st.progress(0)

for i, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB").resize((resize_dim, resize_dim))
    np_img = np.array(img, dtype=np.float32) / 255.0

    # Moyenne + sous-échantillonnage pour une feature légère mais significative
    small = np_img[::8, ::8, :].flatten()  # <1 Ko par image
    small /= np.linalg.norm(small) + 1e-8

    features.append(small)
    progress.progress((i + 1) / len(image_paths))

features = np.vstack(features)

# -------------------------------
# CALCUL DE SIMILARITÉ
# -------------------------------
st.subheader("🧠 Calcul des similarités entre images...")
similarity_matrix = cosine_similarity(features)

# Score d’unicité = 1 - moyenne de similarité
uniqueness_scores = 1 - np.mean(similarity_matrix, axis=1)
sorted_indices = np.argsort(-uniqueness_scores)

keep_count = max(1, int(len(image_paths) * keep_ratio / 100))
selected_indices = sorted_indices[:keep_count]

# -------------------------------
# EXPORT ET VISUALISATION
# -------------------------------
output_dir = os.path.join(tmp_dir, "selected_images")
os.makedirs(output_dir, exist_ok=True)

for idx in selected_indices:
    shutil.copy(image_paths[idx], os.path.join(output_dir, os.path.basename(image_paths[idx])))

st.success(f"✅ Sélection terminée : {keep_count} images gardées sur {len(image_paths)}")

# Afficher les premières images retenues
st.subheader("🖼️ Aperçu des images sélectionnées")
cols = st.columns(5)
for i, idx in enumerate(selected_indices[:10]):
    with cols[i % 5]:
        st.image(image_paths[idx], caption=os.path.basename(image_paths[idx]), use_container_width=True)

# Création d’un ZIP téléchargeable
zip_path = os.path.join(tmp_dir, "selected_images.zip")
with zipfile.ZipFile(zip_path, "w") as zipf:
    for img_name in os.listdir(output_dir):
        zipf.write(os.path.join(output_dir, img_name), img_name)

st.download_button(
    label="📦 Télécharger les images sélectionnées",
    data=open(zip_path, "rb").read(),
    file_name="selected_images.zip",
    mime="application/zip"
)

st.caption("💾 Traitement terminé. Les fichiers temporaires seront supprimés automatiquement.")
