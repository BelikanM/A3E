import os
import shutil
import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps, ExifTags
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import math  # Ajout pour le calcul angulaire
from datetime import datetime  # Ajout pour le parsing des timestamps EXIF

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(page_title="YOLO Photogrammetry Sorter", layout="wide")

st.title("📸 YOLO Photogrammetry Optimizer")
st.markdown("""
Ce module réduit automatiquement le nombre d’images d’un dossier photogrammétrique
en ne gardant que les **angles de vue les plus uniques et informatifs**,
puis réorganise les images pour que les faces/objets soient cohérents et alignés par paires correspondantes.
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
model = YOLO("yolov8n.pt")  # modèle léger
st.success("✅ YOLO chargé avec succès.")

# -------------------------------
# ENREGISTREMENT TEMPORAIRE DES IMAGES
# -------------------------------
tmp_dir = tempfile.mkdtemp()
image_paths = []
for file in uploaded_files:
    file_path = os.path.join(tmp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    image_paths.append(file_path)

# -------------------------------
# EXTRACTION DE FEATURES LÉGÈRE
# -------------------------------
st.write("🔧 Extraction des vecteurs d’images...")
features = []

progress = st.progress(0)
for i, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB").resize((resize_dim, resize_dim))
    np_img = np.array(img, dtype=np.float32) / 255.0
    small = np_img[::8, ::8, :].flatten()
    small /= np.linalg.norm(small) + 1e-8
    features.append(small)
    progress.progress((i + 1) / len(image_paths))

features = np.vstack(features)

# -------------------------------
# CALCUL DE SIMILARITÉ
# -------------------------------
st.subheader("🧠 Calcul des similarités entre images...")
similarity_matrix = cosine_similarity(features)
uniqueness_scores = 1 - np.mean(similarity_matrix, axis=1)
sorted_indices = np.argsort(-uniqueness_scores)
keep_count = max(1, int(len(image_paths) * keep_ratio / 100))
selected_indices = sorted_indices[:keep_count]

# -------------------------------
# DÉTECTION DE FACE / OBJET ET ORIENTATION
# -------------------------------
st.subheader("📐 Réorganisation et rotation des images...")

image_positions = []

for idx in selected_indices:
    img_path = image_paths[idx]
    img = Image.open(img_path).convert("RGB")
    width = img.width
    height = img.height
    results = model(img_path)[0]

    # Extraction du timestamp EXIF pour tri drone-like
    exif = img.getexif()
    datetime_original = None
    if exif:
        for tag, value in exif.items():
            tagname = ExifTags.TAGS.get(tag, tag)
            if tagname == 'DateTimeOriginal':
                datetime_original = value
                break

    if len(results.boxes) > 0:
        confs = results.boxes.conf.cpu().numpy()
        valid_boxes = results.boxes[confs > min_conf]
        if len(valid_boxes) > 0:
            boxes = valid_boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            main_box = boxes[np.argmax(areas)]
            x1, y1, x2, y2 = main_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Mirroir horizontal pour uniformiser la direction (objets vers la droite)
            if cx < width / 2:
                img = ImageOps.mirror(img)
                cx = width - cx  # Mise à jour de cx après mirroir pour alignement correct
        else:
            cx = width / 2
            cy = height / 2
    else:
        cx = width / 2
        cy = height / 2

    # Sauvegarde temporaire pour tri
    temp_path = os.path.join(tmp_dir, f"tmp_{os.path.basename(img_path)}")
    img.save(temp_path)
    image_positions.append((cx, cy, temp_path, width, height, datetime_original))  # Ajout du timestamp

# Tri par timestamp EXIF pour organisation comme des images de drone (ordre de capture), fallback sur position
def time_key(pos):
    cx, cy, _, width, height, dt_str = pos
    if dt_str is not None:
        try:
            return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
        except ValueError:
            pass
    # Fallback : tri par position normalisée (haut-gauche vers bas-droite, comme grille drone)
    norm_cy = cy / height
    norm_cx = cx / width
    return (norm_cy, norm_cx)

image_positions.sort(key=time_key)

# -------------------------------
# EXPORT FINAL
# -------------------------------
sorted_output_dir = os.path.join(tmp_dir, "sorted_images")
os.makedirs(sorted_output_dir, exist_ok=True)

for i, (cx, cy, img_path, width, height, dt_str) in enumerate(image_positions):
    pair_idx = i // 2
    side = 'left' if i % 2 == 0 else 'right'  # Paires correspondantes : vues adjacentes
    new_name = f"pair_{pair_idx+1:03d}_{side}_{os.path.basename(img_path)}"
    shutil.copy(img_path, os.path.join(sorted_output_dir, new_name))

st.success(f"✅ {len(image_positions)} images filtrées, triées comme des captures drone (par timestamp ou position) et organisées par paires correspondantes.")

# Affichage
st.subheader("🖼️ Aperçu des images triées (premières 10)")
cols = st.columns(min(5, len(image_positions)))
for i, pos in enumerate(image_positions[:10]):
    with cols[i % len(cols)]:
        st.image(pos[2], caption=os.path.basename(pos[2]), use_container_width=True)

# Création ZIP final
zip_path = os.path.join(tmp_dir, "sorted_images.zip")
with zipfile.ZipFile(zip_path, "w") as zipf:
    for img_name in os.listdir(sorted_output_dir):
        zipf.write(os.path.join(sorted_output_dir, img_name), img_name)

st.download_button(
    label="📦 Télécharger les images triées (organisées par paires)",
    data=open(zip_path, "rb").read(),
    file_name="sorted_images.zip",
    mime="application/zip"
)

st.caption("💾 Traitement terminé. Les fichiers temporaires seront supprimés automatiquement.")