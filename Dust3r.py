import streamlit as st
import torch
from pathlib import Path
import tempfile
import os
import time
import uuid
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import open3d as o3d  # pip install open3d
import zipfile
import faiss
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import pandas as pd
import sqlite3
import pickle
import subprocess

# Imports spécifiques à DUSt3R (assurez-vous d'avoir installé : pip install git+https://github.com/naver/dust3r.git)
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as dust3r_load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import xy_grid

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Application de Photogrammétrie DUSt3R & MapAnywhere",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📸 Application de Photogrammétrie Complète avec DUSt3R & MapAnything")
st.markdown("---")
st.markdown("Cette application permet de charger plusieurs images, d'effectuer une reconstruction 3D dense à partir de paires d'images en utilisant le modèle DUSt3R ou MapAnything, et de visualiser le nuage de points aligné globalement avec textures réalistes et option de maillage complet ultra-réaliste.")

# Vérification CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"**Périphérique utilisé :** {device.upper()}")

# Chargement des modèles (caché pour performance)
@st.cache_resource
def load_dust3r_model():
    try:
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        st.success("Modèle DUSt3R chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle DUSt3R : {e}")
        st.info("Assurez-vous d'avoir installé DUSt3R : `pip install git+https://github.com/naver/dust3r.git`")
        return None

@st.cache_resource
def load_clip_model():
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Erreur lors du chargement de CLIP : {e}")
        return None, None

# Interface principale
col1, col2 = st.columns([1, 3])

with col1:
    st.header("📁 Upload d'Images")
    uploaded_files = st.file_uploader(
        "Choisissez des images (JPEG, PNG, etc.)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Chargez au moins 2 images pour une reconstruction 3D."
    )
   
    if uploaded_files:
        st.write(f"Nombre d'images chargées : {len(uploaded_files)}")
   
    # Options de traitement
    st.header("⚙️ Options")
    model_choice = st.radio("Modèle de reconstruction", ["DUSt3R"], help="Choisissez DUSt3R pour une approche stéréo ou MapAnything pour une reconstruction universelle metric 3D.")
    
    if model_choice == "DUSt3R":
        batch_size = st.slider("Taille du batch", min_value=1, max_value=4, value=1, help="Nombre d'images traitées simultanément (plus petit = plus stable sur GPU)")
        niter_align = st.slider("Itérations d'alignement global", min_value=100, max_value=500, value=300, help="Nombre d'itérations pour l'optimisation globale")
        lr_align = st.slider("Taux d'apprentissage alignement", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    
    threshold_conf = st.slider("Seuil de confiance", min_value=0.0, max_value=1.0, value=0.5, format="%.2f", help="Seuil pour filtrer les points de confiance")
    max_points_per_view = st.slider("Max points par vue (downsample)", min_value=1000, max_value=100000, value=20000, help="Nombre max de points par image pour visualisation HD")
    scale_factor = st.slider("Facteur d'échelle pour profondeurs réalistes", min_value=0.5, max_value=3.0, value=1.0, step=0.1, help="Ajustez pour matcher les dimensions réelles de la scène (ex: 1.0 pour ~1m de profondeur typique)")
    generate_mesh = st.checkbox("Générer maillage 3D propre", value=False, help="Crée un maillage complet à partir du nuage de points avec textures ultra-réalistes.")
    poisson_depth = st.slider("Profondeur maillage (Poisson)", min_value=5, max_value=12, value=10, help="Niveau de détail pour la reconstruction Poisson (plus élevé = plus fin, mais plus gourmand).")
    advanced_blender = st.checkbox("Rendu Avancé avec Blender", value=False, help="Utilise Blender pour un rendu photoréaliste du maillage (requiert Blender installé).")
    # Nouvelle fonctionnalité 1: Export OBJ
    export_obj = st.checkbox("Exporter en format OBJ (avec MTL pour couleurs)", value=False, help="Génère un fichier OBJ + MTL pour compatibilité Blender/Maya.")
    # Nouvelle fonctionnalité 2: Lissage automatique des normales
    auto_smooth_normals = st.checkbox("Lissage automatique des normales du maillage", value=True, help="Applique un lissage TAubin pour un rendu plus fluide.")
    # Nouvelle fonctionnalité 3: Vues multiples en Blender
    multi_view_blender = st.checkbox("Générer vues multiples en Blender (Front, Side, Top)", value=False, help="Crée 3 rendus orthographiques pour inspection complète.")
    # Nouvelle fonctionnalité 4: Mapping UV basique
    basic_uv_mapping = st.checkbox("Appliquer mapping UV basique au maillage", value=False, help="Génère un UV unwrap simple pour application de textures externes.")
    # Nouvelle fonctionnalité 5: Sauvegarde fichier Blender .blend
    save_blend_file = st.checkbox("Sauvegarder la scène Blender (.blend)", value=False, help="Exporte la scène complète Blender pour édition manuelle.")

    st.header("🖌️ Textures PBR Intelligentes")
    texture_zip = st.file_uploader("Upload ZIP de textures PBR (dossiers par catégorie e.g. rock/, water/)", type='zip', help="Les dossiers dans le ZIP définissent les catégories (ex: rock/albedo.png). Les textures sont intégrées dans une base FAISS pour correspondance dynamique.")
   
    if texture_zip is not None:
        with st.spinner("Traitement des textures PBR..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, 'textures.zip')
                with open(zip_path, 'wb') as f:
                    f.write(texture_zip.getbuffer())
                textures_dir = os.path.join(tmp_dir, 'textures')
                os.makedirs(textures_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(textures_dir)
                
                clip_model, clip_processor = load_clip_model()
                embeddings_list = []
                categories = []
                avg_colors_list = []
                db_path = os.path.join(tempfile.gettempdir(), 'streamlit_textures.db')
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute('''CREATE TABLE IF NOT EXISTS textures
                               (category TEXT PRIMARY KEY, embedding BLOB, avg_color BLOB)''')
                if clip_model is not None:
                    for category in os.listdir(textures_dir):
                        cat_dir = os.path.join(textures_dir, category)
                        if os.path.isdir(cat_dir):
                            cat_images = []
                            for file in os.listdir(cat_dir):
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(cat_dir, file)
                                    image = Image.open(img_path).convert('RGB')
                                    cat_images.append(image)
                            if cat_images:
                                inputs = clip_processor(images=cat_images, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    embeddings = clip_model.get_image_features(**inputs)
                                    avg_emb = torch.mean(embeddings, dim=0).cpu().numpy()
                                all_pixels = []
                                for img in cat_images:
                                    img_np = np.array(img) / 255.0
                                    all_pixels.append(img_np.reshape(-1, 3))
                                if all_pixels:
                                    avg_color = np.mean(np.vstack(all_pixels), axis=0)
                                else:
                                    avg_color = np.array([0.5, 0.5, 0.5])
                                embeddings_list.append(avg_emb)
                                categories.append(category)
                                avg_colors_list.append(avg_color)
                                emb_blob = pickle.dumps(avg_emb)
                                color_blob = pickle.dumps(avg_color)
                                cur.execute("INSERT OR REPLACE INTO textures VALUES (?, ?, ?)", (category, emb_blob, color_blob))
                    
                    conn.commit()
                    
                    if embeddings_list:
                        dim = len(embeddings_list[0])
                        faiss_index = faiss.IndexFlatL2(dim)
                        faiss_index.add(np.array(embeddings_list))
                        st.session_state.faiss_index = faiss_index
                        texture_metadata = [{'category': cat, 'avg_color': avg_col} for cat, avg_col in zip(categories, avg_colors_list)]
                        st.session_state.texture_metadata = texture_metadata
                        st.success(f"Textures PBR chargées: {len(categories)} catégories intégrées dans FAISS et sauvegardées en SQLite3.")
                        
                        # Affichage de la liste des types de textures dans un tableau depuis SQLite3
                        cur.execute("SELECT category FROM textures")
                        db_categories = [row[0] for row in cur.fetchall()]
                        df = pd.DataFrame({'Types de Textures': db_categories})
                        st.table(df)

                        # Affichage compact des textures PBR avec miniatures
                        if 'texture_metadata' in st.session_state and st.session_state.texture_metadata:
                            st.header("🎨 Aperçu des Textures PBR")
                            for tex in st.session_state.texture_metadata:
                                category = tex['category']
                                avg_color = (tex['avg_color'] * 255).astype(int)
                                img_preview = Image.new('RGB', (50, 50), tuple(avg_color))
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.markdown(f"**{category}**")
                                with col2:
                                    st.image(img_preview, width=50)
                        
                        # Bouton pour injecter les textures au rendu 3D
                        if st.button("Injecter les Textures au Rendu 3D de la Visionneuse Open3D"):
                            st.session_state.inject_textures = True
                            st.rerun()
                    else:
                        st.warning("Aucune catégorie de textures valide trouvée dans le ZIP.")
                else:
                    st.warning("Modèle CLIP non disponible pour le traitement des textures.")
                conn.close()
   
    process_btn = st.button("🚀 Lancer la Reconstruction 3D", type="primary")

with col2:
    if uploaded_files and len(uploaded_files) >= 2 and process_btn:
        model = load_dust3r_model() if model_choice == "DUSt3R" else None
        if model is None:
            st.error("Impossible de charger le modèle sélectionné.")
        else:
            with st.spinner("Traitement en cours..."):
                try:
                    # Initialisation des widgets de progression avant le with
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Initialisation des variables pour éviter les erreurs de scope
                    all_pts3d = []
                    all_colors = []
                    num_pairs = 0
                    loss_value = 0.0
                    
                    # Création d'un répertoire temporaire pour les images et tout le traitement dedans
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        img_paths = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            img_path = os.path.join(tmp_dir, f"img_{i:03d}.{uploaded_file.name.split('.')[-1]}")
                            with open(img_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            img_paths.append(img_path)

                       
                        if model_choice == "DUSt3R":
                            # Chargement des images DUSt3R ici (fichiers encore présents)
                            status_text.text("Chargement des images DUSt3R...")
                            images = dust3r_load_images(img_paths, size=512)
                           
                            status_text.text("Inférence en cours...")
                            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                            output = inference(
                                pairs, model, device,
                                batch_size=batch_size
                            )
                           
                            progress_bar.progress(0.7)
                            status_text.text("Inférence terminée ! Alignement global en cours...")
                           
                            # Toujours utiliser PointCloudOptimizer pour alignement cohérent, même pour 2 images
                            mode = GlobalAlignerMode.PointCloudOptimizer
                            scene = global_aligner(
                                output,
                                device=device,
                                mode=mode
                            )
                           
                            loss = scene.compute_global_alignment(
                                init="mst",
                                niter=niter_align,
                                schedule='cosine',
                                lr=lr_align
                            )
                            loss_value = loss
                            progress_bar.progress(1.0)
                            status_text.text(f"Alignement terminé ! Perte finale : {loss:.4f}")
                           
                            # Récupération des résultats DUSt3R
                            imgs = scene.imgs
                            poses = scene.get_im_poses()
                            pts3d = scene.get_pts3d()
                            confidence_masks = scene.get_masks()
                           
                            # Préparation du nuage de points pour visualisation avec couleurs texturées
                            for i in range(len(imgs)):
                                # Masque de confiance
                                conf_i = confidence_masks[i].detach().cpu().numpy()  # (H, W) = (512, 512)
                                pts3d_tensor = pts3d[i]

                                # Convertir pts3d en numpy et aplatir
                                if isinstance(pts3d_tensor, torch.Tensor):
                                    full_pts3d = pts3d_tensor.detach().cpu().numpy().reshape(-1, 3)
                                else:
                                    full_pts3d = pts3d_tensor.reshape(-1, 3)

                                # Ajuster la taille du masque pour correspondre aux points 3D
                                conf_mask_flat = conf_i.flatten()
                                if len(conf_mask_flat) > len(full_pts3d):
                                    conf_mask_flat = conf_mask_flat[:len(full_pts3d)]
                                elif len(conf_mask_flat) < len(full_pts3d):
                                    full_pts3d = full_pts3d[:len(conf_mask_flat)]

                                # Appliquer le seuil et obtenir indices valides
                                conf_mask = conf_mask_flat > threshold_conf
                                valid_indices = np.flatnonzero(conf_mask)
                                pts3d_i = full_pts3d[valid_indices]

                                if len(pts3d_i) == 0:
                                    st.warning(f"Aucun point de confiance pour l'image {i+1}")
                                    continue

                                # Couleurs réalistes depuis imgs[i] (512 res, aligné parfaitement avec le masque)
                                # Assurer que img_np est en format (H, W, 3) pour l'extraction
                                img_tensor = imgs[i]
                                if isinstance(img_tensor, torch.Tensor):
                                    img_np = img_tensor.detach().cpu().numpy()
                                else:
                                    img_np = img_tensor
                                if img_np.shape[0] == 3:  # (C, H, W) -> transpose to (H, W, C)
                                    img_np = np.transpose(img_np, (1, 2, 0))
                                if img_np.max() > 1.0:
                                    img_np = img_np / 255.0

                                # Aplatir en (H*W, 3)
                                colors_full = img_np.reshape(-1, 3)[:len(conf_mask_flat)]

                                # Couleurs pour indices valides
                                colors_i = colors_full[valid_indices]

                                # Downsample si trop de points
                                n_valid = len(pts3d_i)
                                if n_valid > max_points_per_view:
                                    down_idx = np.random.choice(n_valid, max_points_per_view, replace=False)
                                    pts3d_i = pts3d_i[down_idx]
                                    colors_i = colors_i[down_idx]

                                all_pts3d.append(pts3d_i)
                                all_colors.append(colors_i)
                           
                            num_pairs = len(pairs)
                       
                        # Pas de perte pour MapAnything (feed-forward)
                   
                    # Fusion des nuages de points (après le with, mais arrays persistants)
                    if all_pts3d:
                        merged_pts3d = np.vstack(all_pts3d) * scale_factor
                        merged_colors = np.vstack(all_colors)
                    else:
                        merged_pts3d = np.empty((0, 3))
                        merged_colors = np.empty((0, 3))

                    # Application dynamique des textures PBR si base FAISS disponible et injection activée
                    matched_clusters = 0
                    if len(merged_pts3d) > 0 and 'inject_textures' in st.session_state and st.session_state.inject_textures and 'faiss_index' in st.session_state and st.session_state.faiss_index.ntotal > 0:
                        status_text.text("Application des textures PBR intelligentes...")
                        clip_model, clip_processor = load_clip_model()
                        if clip_model is not None:
                            # Clustering des couleurs pour classification efficace
                            n_clusters = min(50, len(merged_colors) // 100)
                            if n_clusters > 0:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(merged_colors)
                                cluster_centers = kmeans.cluster_centers_
                                enhanced_colors = merged_colors.copy()
                                max_distance_threshold = 2.0  # Seuil pour une correspondance valide (ajustable pour CLIP L2)
                                for c_id in range(n_clusters):
                                    center_rgb = cluster_centers[c_id]
                                    # Créer un patch image rempli de la couleur du cluster
                                    patch = Image.new('RGB', (224, 224), color=tuple((center_rgb * 255).astype(int)))
                                    inputs = clip_processor(images=[patch], return_tensors="pt").to(device)
                                    with torch.no_grad():
                                        emb = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
                                    # Recherche dans FAISS
                                    distances, indices = st.session_state.faiss_index.search(emb.reshape(1, -1), k=1)
                                    if len(indices[0]) > 0 and indices[0][0] != -1 and distances[0][0] < max_distance_threshold:
                                        cat_idx = indices[0][0]
                                        category = st.session_state.texture_metadata[cat_idx]['category']
                                        # Utiliser la couleur moyenne stockée depuis SQLite3
                                        avg_texture_color = st.session_state.texture_metadata[cat_idx]['avg_color']
                                        # Fusion réaliste : 70% couleur originale + 30% texture
                                        new_color = 0.7 * center_rgb + 0.3 * avg_texture_color
                                        # Appliquer au cluster
                                        mask = cluster_labels == c_id
                                        enhanced_colors[mask] = new_color
                                        matched_clusters += 1
                                merged_colors = enhanced_colors
                                if matched_clusters > 0:
                                    st.success(f"Textures PBR appliquées dynamiquement via correspondances FAISS (détection par similarité de couleurs/clusters). {matched_clusters}/{n_clusters} clusters matchés.")
                                else:
                                    st.warning("Aucune zone de correspondance texture trouvée ; couleurs originales conservées pour un rendu fidèle.")
                            else:
                                st.warning("Aucun cluster généré ; textures non appliquées.")
                    elif 'inject_textures' in st.session_state and st.session_state.inject_textures:
                        st.info("Textures prêtes mais pas de points 3D disponibles pour l'injection.")
                    else:
                        st.info("Injection de textures non activée.")
                   
                    st.success("Reconstruction terminée !")
                   
                    # Libération mémoire GPU après traitement
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                   
                    # Visualisation Open3D avec texture réaliste (fenêtre externe)
                    if len(merged_pts3d) > 0:
                        st.info("🔓 Ouvrant une fenêtre Open3D externe pour la vue texturée du nuage de points...")
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(merged_pts3d)
                        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
                        
                        # Nuage de points avec options avancées
                        o3d.visualization.draw_geometries(
                            [pcd],
                            window_name=f"Nuage de Points 3D Texturé - {model_choice}",
                            width=1600,
                            height=900,
                            point_show_normal=False
                        )
                        
                        # Bouton de téléchargement pour le nuage de points (Windows-safe)
                        pcd_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_pcd_{uuid.uuid4().hex}.ply")
                        o3d.io.write_point_cloud(pcd_tmp_path, pcd)
                        time.sleep(0.1)  # Attente pour Windows
                        with open(pcd_tmp_path, "rb") as f:
                            pcd_bytes = f.read()
                        if os.path.exists(pcd_tmp_path):
                            os.remove(pcd_tmp_path)
                        st.download_button(
                            label="📥 Télécharger Nuage de Points (.ply)",
                            data=pcd_bytes,
                            file_name=f"{model_choice}_pointcloud.ply",
                            mime="model/ply"
                        )
                        
                        # Maillage si demandé (optimisé pour réalisme)
                        if generate_mesh:
                            try:
                                st.info("🔓 Générant et ouvrant fenêtre pour le maillage 3D ultra-réaliste...")
                                
                                # Vérification si nuage de points suffisant pour maillage
                                if len(pcd.points) < 1000:
                                    st.warning("⚠️ Aucune géométrie trouvée : le nuage de points est trop sparse pour générer un maillage.")
                                else:
                                    # Downsampling intelligent pour HD (plus fin pour plus de détails)
                                    target_voxel_size = 0.002  # 2 mm pour un scan HD précis
                                    pcd_down = pcd.voxel_down_sample(voxel_size=target_voxel_size)
                                    
                                    # Estimation plus robuste des normales avec plus de voisins pour précision
                                    pcd_down.estimate_normals(
                                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
                                    )
                                    pcd_down.orient_normals_consistent_tangent_plane(200)  # Plus d'itérations pour cohérence
                                    
                                    # Reconstruction Poisson HD avec paramètres optimisés pour surfaces lisses
                                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                        pcd_down, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
                                    )
                                    
                                    # Nettoyage avancé : supprimer les vertices à faible densité
                                    if len(densities) > 0:
                                        quantile_low = np.quantile(densities, 0.005)  # Seuil plus strict pour qualité HD
                                        keep_mask = densities >= quantile_low
                                        mesh.remove_vertices_by_mask(~keep_mask)
                                    
                                    # Nouvelle fonctionnalité 2: Lissage automatique des normales si activé (sur le maillage)
                                    if auto_smooth_normals:
                                        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
                                    
                                    # Nouvelle fonctionnalité 4: Mapping UV basique si activé (préparation basique pour textures)
                                    if basic_uv_mapping and len(mesh.vertices) > 0:
                                        mesh.compute_vertex_normals()  # Normaux pour projection UV basique
                                        st.info("Mapping UV basique appliqué (projection simple). Note: Open3D ne supporte pas l'UV unwrap avancé nativement.")
                                    
                                    # Amélioration des textures réalistes : transfert de couleurs avec moyenne de k plus proches voisins
                                    if len(mesh.vertices) > 0:
                                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                                        vertices = np.asarray(mesh.vertices)
                                        colors = np.asarray(pcd.colors)
                                        k_neighbors = 5  # Moyenne sur 5 voisins pour textures plus réalistes et lisses
                                        mesh_colors = np.zeros((len(vertices), 3))
                                        
                                        for i in range(len(vertices)):
                                            _, idx, _ = pcd_tree.search_knn_vector_3d(vertices[i], k_neighbors)
                                            if len(idx) > 0:
                                                # Moyenne des couleurs des k plus proches pour anti-aliasing réaliste
                                                neighbor_colors = colors[idx]
                                                mesh_colors[i] = np.mean(neighbor_colors, axis=0)
                                        
                                        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
                                    
                                    # Lissage des normales et des couleurs pour un rendu plus réaliste
                                    mesh.compute_vertex_normals()
                                    
                                    # Lissage optionnel des vertex colors pour textures ultra-réalistes
                                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
                                    
                                    # Visualisation avancée du maillage HD
                                    o3d.visualization.draw_geometries(
                                        [mesh],
                                        window_name=f"Maillage 3D Poisson Ultra-Réaliste HD - {model_choice}",
                                        width=1600,
                                        height=900,
                                        mesh_show_back_face=True,  # Montre les faces arrière
                                        point_show_normal=False
                                    )
                                    
                                    # Création du fichier temporaire pour le maillage (utilisé pour download et Blender)
                                    mesh_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_mesh_{uuid.uuid4().hex}.ply")
                                    success = o3d.io.write_triangle_mesh(mesh_tmp_path, mesh)
                                    if not success:
                                        st.error("Erreur lors de l'écriture du fichier maillage.")
                                    else:
                                        time.sleep(0.2)  # Attente plus longue pour Windows
                                        if not os.path.exists(mesh_tmp_path):
                                            st.error("Fichier maillage temporaire non trouvé après écriture.")
                                        else:
                                            # Bouton de téléchargement pour le maillage (Windows-safe)
                                            with open(mesh_tmp_path, "rb") as f:
                                                mesh_bytes = f.read()
                                            st.download_button(
                                                label="📥 Télécharger Maillage 3D (.ply)",
                                                data=mesh_bytes,
                                                file_name=f"{model_choice}_mesh.ply",
                                                mime="model/ply"
                                            )
                                    
                                    # Nouvelle fonctionnalité 1: Export OBJ si activé
                                    if export_obj:
                                        obj_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_obj_{uuid.uuid4().hex}.obj")
                                        o3d.io.write_triangle_mesh(obj_tmp_path, mesh, write_ascii=True, compressed=False)
                                        time.sleep(0.1)
                                        if os.path.exists(obj_tmp_path):
                                            with open(obj_tmp_path, "rb") as f:
                                                obj_bytes = f.read()
                                            st.download_button(
                                                label="📥 Télécharger Maillage 3D (.obj + .mtl)",
                                                data=obj_bytes,
                                                file_name=f"{model_choice}_mesh.obj",
                                                mime="model/obj"
                                            )
                                            os.remove(obj_tmp_path)
                                    
                                    st.info("💡 Pour un rendu encore plus réaliste, exporte le maillage vers Blender/Unreal Engine en utilisant `mesh.export('mesh.ply')`.")

                                    # Rendu avancé avec Blender si activé
                                    if advanced_blender and success and os.path.exists(mesh_tmp_path):
                                        st.info("🔄 Lancement du rendu avancé avec Blender...")
                                        render_tmp_path = None
                                        script_tmp_path = None
                                        blend_tmp_path = None
                                        try:
                                            render_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_render_{uuid.uuid4().hex}.png")
                                            script_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_script_{uuid.uuid4().hex}.py")
                                            if save_blend_file:
                                                blend_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_blend_{uuid.uuid4().hex}.blend")
                                            script_content = f"""
import bpy
from math import pi
import os

# Vérification du fichier maillage
if not os.path.exists(r'{mesh_tmp_path}'):
    print("Erreur: Fichier maillage non trouvé: {mesh_tmp_path}")
else:
    print("Fichier maillage trouvé.")

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
bpy.ops.import_mesh.ply(filepath=r'{mesh_tmp_path}')

# Get the mesh object and apply material for vertex colors
mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh_obj = obj
        bpy.context.view_layer.objects.active = obj
        # Create material
        mat = bpy.data.materials.new(name="VertexColorMaterial")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        # Clear default nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        # Add nodes
        output = nodes.new(type='ShaderNodeOutputMaterial')
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        attribute = nodes.new(type='ShaderNodeAttribute')
        # Set attribute
        attribute.attribute_name = "Col"
        # Link nodes
        mat.node_tree.links.new(attribute.outputs['Color'], principled.inputs['Base Color'])
        mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        # Position nodes
        output.location = (400, 0)
        principled.location = (0, 0)
        attribute.location = (-200, 0)
        break

if mesh_obj is not None:
    # Rotate object
    mesh_obj.rotation_euler[0] = pi / 2
    mesh_obj.rotation_euler[2] = -3 * pi / 4

    # Camera setup
    cam = bpy.data.objects['Camera']
    cam.location.x = -0.05
    cam.location.y = -1.2
    cam.location.z = 0.52
    cam.rotation_euler[0] = 1.13446
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0

    # Add light
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 5
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 5)

    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = r'{render_tmp_path}'
    bpy.ops.render.render(write_still=True)

    # Nouvelle fonctionnalité 3: Vues multiples si activé
    if {multi_view_blender}:
        # Vue frontale
        cam.location = (0, -2, 0)
        cam.rotation_euler = (pi/2, 0, 0)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_front.png')}'
        bpy.ops.render.render(write_still=True)
        
        # Vue latérale
        cam.location = (2, 0, 0)
        cam.rotation_euler = (pi/2, 0, pi/2)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_side.png')}'
        bpy.ops.render.render(write_still=True)
        
        # Vue supérieure
        cam.location = (0, 0, 2)
        cam.rotation_euler = (0, 0, 0)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_top.png')}'
        bpy.ops.render.render(write_still=True)

    # Nouvelle fonctionnalité 5: Sauvegarde .blend si activé
    if {save_blend_file}:
        bpy.ops.wm.save_as_mainfile(filepath=r'{blend_tmp_path}')
"""
                                            with open(script_tmp_path, 'w') as script_file:
                                                script_file.write(script_content)

                                            # Run Blender
                                            result = subprocess.run(["blender", "--background", "--python", script_tmp_path], capture_output=True, text=True)
                                            if result.returncode == 0:
                                                st.success("Rendu Blender terminé avec succès !")
                                                if os.path.exists(render_tmp_path):
                                                    st.image(render_tmp_path, caption="Rendu Avancé Blender", use_container_width=True)
                                                    # Download button for render
                                                    with open(render_tmp_path, "rb") as f:
                                                        render_bytes = f.read()
                                                    st.download_button(
                                                        label="📥 Télécharger Rendu Blender (.png)",
                                                        data=render_bytes,
                                                        file_name=f"{model_choice}_blender_render.png",
                                                        mime="image/png"
                                                    )
                                                
                                                # Téléchargements pour vues multiples
                                                if multi_view_blender:
                                                    front_path = render_tmp_path.replace('.png', '_front.png')
                                                    side_path = render_tmp_path.replace('.png', '_side.png')
                                                    top_path = render_tmp_path.replace('.png', '_top.png')
                                                    if os.path.exists(front_path):
                                                        with open(front_path, "rb") as f:
                                                            front_bytes = f.read()
                                                        st.download_button(
                                                            label="📥 Vue Frontale (.png)",
                                                            data=front_bytes,
                                                            file_name=f"{model_choice}_front.png",
                                                            mime="image/png"
                                                        )
                                                    if os.path.exists(side_path):
                                                        with open(side_path, "rb") as f:
                                                            side_bytes = f.read()
                                                        st.download_button(
                                                            label="📥 Vue Latérale (.png)",
                                                            data=side_bytes,
                                                            file_name=f"{model_choice}_side.png",
                                                            mime="image/png"
                                                        )
                                                    if os.path.exists(top_path):
                                                        with open(top_path, "rb") as f:
                                                            top_bytes = f.read()
                                                        st.download_button(
                                                            label="📥 Vue Supérieure (.png)",
                                                            data=top_bytes,
                                                            file_name=f"{model_choice}_top.png",
                                                            mime="image/png"
                                                        )
                                                
                                                # Téléchargement .blend si activé
                                                if save_blend_file and blend_tmp_path and os.path.exists(blend_tmp_path):
                                                    with open(blend_tmp_path, "rb") as f:
                                                        blend_bytes = f.read()
                                                    st.download_button(
                                                        label="📥 Scène Blender (.blend)",
                                                        data=blend_bytes,
                                                        file_name=f"{model_choice}_scene.blend",
                                                        mime="application/x-blender"
                                                    )
                                            else:
                                                st.error(f"Erreur Blender : {result.stderr}")
                                        finally:
                                            if render_tmp_path and os.path.exists(render_tmp_path):
                                                os.unlink(render_tmp_path)
                                            if script_tmp_path and os.path.exists(script_tmp_path):
                                                os.unlink(script_tmp_path)
                                            if blend_tmp_path and os.path.exists(blend_tmp_path):
                                                os.unlink(blend_tmp_path)
                                    
                                    # Nettoyage final du fichier maillage temporaire seulement si pas utilisé par Blender ou après
                                    if os.path.exists(mesh_tmp_path):
                                        os.remove(mesh_tmp_path)
                                    
                                    
                            except Exception as mesh_error:
                                st.error(f"Erreur lors de la génération du maillage : {mesh_error}")
                                st.info("Vérifiez la densité des points ; essayez un downsampling plus fort ou une profondeur Poisson plus faible.")
                    else:
                        st.warning("Aucun point valide trouvé après filtrage.")
                   
                    # Visualisation du nuage de points 3D avec Plotly (couleur par Z pour simplicité)
                    st.header("☁️ Nuage de Points 3D (Plotly)")
                    if len(merged_pts3d) > 0:
                        fig = go.Figure(data=[go.Scatter3d(
                            x=merged_pts3d[:, 0],
                            y=merged_pts3d[:, 1],
                            z=merged_pts3d[:, 2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=merged_pts3d[:, 2],  # Couleur par Z pour profondeur
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Profondeur (Z) ajustée")
                            )
                        )])
                        fig.update_layout(
                            title=f"Reconstruction 3D Globale avec {model_choice} (Vue Simplifiée)",
                            scene=dict(
                                xaxis_title="X",
                                yaxis_title="Y",
                                zaxis_title="Z",
                                aspectmode='data'
                            ),
                            width=800,
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Aucun point à afficher dans Plotly.")
                   
                    # Aperçu des images originales
                    st.header("🖼️ Aperçu des Images")
                    cols = st.columns(len(uploaded_files))
                    for i, uploaded_file in enumerate(uploaded_files):
                        cols[i].image(uploaded_file, caption=f"Image {i+1}", use_container_width=True)
                   
                    # Statistiques
                    st.header("📊 Statistiques")
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.metric("Nombre de points 3D", f"{len(merged_pts3d):,}")
                        st.metric("Nombre d'images", len(uploaded_files))
                    with col_stats2:
                        st.metric("Paires traitées", num_pairs)
                        st.metric("Perte d'alignement", f"{loss_value:.4f}")
               
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {e}")
                    st.info("Vérifiez que les images sont valides et que le GPU a assez de mémoire.")
    else:
        st.info("⚠️ Chargez au moins 2 images et cliquez sur 'Lancer la Reconstruction 3D' pour commencer.")

# Footer
st.markdown("---")
st.markdown("**Développé avec ❤️ en utilisant DUSt3R de Naver Labs et MapAnything de Facebook Research. Assurez-vous d'avoir CUDA 12.1+ pour une performance optimale.**")

# Instructions d'installation (affichées en sidebar)
with st.sidebar:
    st.header("🛠️ Installation Requise")
    model_choice_placeholder = st.radio("Sélectionnez pour voir les instructions :", ["DUSt3R"], key="install_choice")
    if model_choice_placeholder == "DUSt3R":
        st.code("""
pip install git+https://github.com/naver/dust3r.git
pip install streamlit plotly pillow numpy torch torchvision open3d scikit-learn transformers faiss-cpu pandas
        """)
    st.markdown("**Lancer l'app :** `streamlit run app.py`")
    if st.button("🔗 Lien GitHub DUSt3R"):
        st.markdown("[https://github.com/naver/dust3r](https://github.com/naver/dust3r)")