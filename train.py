import os
import cv2
import numpy as np
import streamlit as st
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import shutil
import zipfile

# --- CONFIGURATION ---
st.set_page_config(page_title="🛰️ Sélecteur d’Images Drone", layout="wide")
st.title("🛰️ Sélecteur intelligent d’images drone")

# --- PARAMÈTRES UTILISATEUR ---
MIN_MATCHES = st.sidebar.slider("🔗 Nombre minimal de correspondances", 20, 300, 80)
STEP = st.sidebar.slider("↔️ Saut entre images", 1, 10, 3)

uploaded_files = st.file_uploader(
    "📤 Dépose ici tes images drone (jpg/png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("⚠️ Merci d’uploader quelques images de drone pour commencer.")
    st.stop()

# --- Dossier temporaire ---
temp_dir = tempfile.mkdtemp()
for file in uploaded_files:
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

# --- Extraction de features ---
@st.cache_data(show_spinner=False)
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    orb = cv2.ORB_create(2000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


# --- Matching entre deux images ---
def match_images(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len([m for m in matches if m.distance < 50])


# --- Analyse principale ---
@st.cache_data(show_spinner=True)
def analyze_folder(folder, step, min_matches):
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    st.info(f"Extraction des features de {len(images)} images...")
    features = {}
    for img_path in tqdm(images):
        kp, desc = extract_features(img_path)
        if desc is not None:
            features[img_path] = desc

    st.info("Calcul des correspondances entre images...")
    connectivity = defaultdict(set)
    for i, img1 in enumerate(tqdm(images)):
        if img1 not in features:
            continue
        for j in range(i + 1, min(i + step + 1, len(images))):
            img2 = images[j]
            if img2 not in features:
                continue
            n_matches = match_images(features[img1], features[img2])
            if n_matches > min_matches:
                connectivity[img1].add(img2)
                connectivity[img2].add(img1)
    return connectivity


# --- Sélection d’images utiles ---
def select_minimal_set(connectivity):
    selected = set()
    visited = set()
    for img, neighbors in connectivity.items():
        if img not in visited:
            selected.add(img)
            visited.update(neighbors)
            visited.add(img)
    return selected


# --- Visualisation graphe ---
def show_graph(connectivity):
    G = nx.Graph()
    for img, neighbors in connectivity.items():
        for n in neighbors:
            G.add_edge(os.path.basename(img), os.path.basename(n))
    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=False, node_size=60, width=0.6)
    st.pyplot(plt)


# --- Lancement de l’analyse ---
if st.button("🚀 Lancer l’analyse"):
    with st.spinner("Analyse en cours..."):
        connectivity = analyze_folder(temp_dir, STEP, MIN_MATCHES)
        selected = select_minimal_set(connectivity)

    st.success(f"✅ {len(selected)} images retenues sur {len(uploaded_files)}")
    
    # Graphe
    st.subheader("🔗 Graphe des connexions détectées")
    show_graph(connectivity)

    # Aperçu images sélectionnées
    st.subheader("📸 Images sélectionnées")
    cols = st.columns(5)
    for i, img_path in enumerate(selected):
        with cols[i % 5]:
            st.image(img_path, use_container_width=True)

    # Créer un ZIP à télécharger
    zip_path = os.path.join(temp_dir, "selected_images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img in selected:
            zipf.write(img, os.path.basename(img))
    
    with open(zip_path, "rb") as f:
        st.download_button("⬇️ Télécharger les images sélectionnées (ZIP)", f, "selected_images.zip")
