import streamlit as st
import torch
from pathlib import Path
import tempfile
import os
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
    page_title="Application de Photogrammétrie DUSt3R & MapAnything",
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

    st.header("🖌️ Textures PBR Intelligentes")
    texture_zip = st.file_uploader("Upload ZIP de textures PBR (dossiers par catégorie e.g. rock/, water/)", type='zip', help="Les dossiers dans le ZIP définissent les catégories (ex: rock/albedo.png). Les textures sont intégrées dans une base FAISS pour correspondance dynamique.")
   
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
                    
                    # Création d'un répertoire temporaire pour les images et tout le traitement dedans
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        img_paths = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            img_path = os.path.join(tmp_dir, f"img_{i:03d}.{uploaded_file.name.split('.')[-1]}")
                            with open(img_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            img_paths.append(img_path)

                        # Traitement des textures PBR si uploadées
                        faiss_index = None
                        texture_metadata = []
                        if texture_zip is not None:
                            status_text.text("Traitement des textures PBR...")
                            zip_path = os.path.join(tmp_dir, 'textures.zip')
                            with open(zip_path, 'wb') as f:
                                f.write(texture_zip.getbuffer())
                            textures_dir = os.path.join(tmp_dir, 'textures')
                            os.makedirs(textures_dir, exist_ok=True)
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                z.extractall(textures_dir)
                            
                            clip_model, clip_processor = load_clip_model()
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
                                            texture_metadata.append({'category': category, 'embedding': avg_emb, 'images': cat_images})
                                
                                if texture_metadata:
                                    dim = len(texture_metadata[0]['embedding'])
                                    faiss_index = faiss.IndexFlatL2(dim)
                                    embeddings_list = [meta['embedding'] for meta in texture_metadata]
                                    faiss_index.add(np.array(embeddings_list))
                                    st.session_state.faiss_index = faiss_index
                                    st.session_state.texture_metadata = texture_metadata
                                    st.success(f"Textures PBR chargées: {len(texture_metadata)} catégories intégrées dans FAISS.")
                            progress_bar.progress(0.1)
                       
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
                           
                            loss_value = 0.0
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
                            all_pts3d = []
                            all_colors = []
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
                       
                        loss_value = 0.0  # Pas de perte pour MapAnything (feed-forward)
                   
                    # Fusion des nuages de points (après le with, mais arrays persistants)
                    if all_pts3d:
                        merged_pts3d = np.vstack(all_pts3d) * scale_factor
                        merged_colors = np.vstack(all_colors)
                    else:
                        merged_pts3d = np.empty((0, 3))
                        merged_colors = np.empty((0, 3))

                    # Application dynamique des textures PBR si base FAISS disponible
                    if len(merged_pts3d) > 0 and 'faiss_index' in st.session_state and st.session_state.faiss_index.ntotal > 0:
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
                                for c_id in range(n_clusters):
                                    center_rgb = cluster_centers[c_id]
                                    # Créer un patch image rempli de la couleur du cluster
                                    patch = Image.new('RGB', (224, 224), color=tuple((center_rgb * 255).astype(int)))
                                    inputs = clip_processor(images=[patch], return_tensors="pt").to(device)
                                    with torch.no_grad():
                                        emb = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
                                    # Recherche dans FAISS
                                    distances, indices = st.session_state.faiss_index.search(emb.reshape(1, -1), k=1)
                                    if len(indices[0]) > 0 and indices[0][0] != -1:
                                        cat_idx = indices[0][0]
                                        category = st.session_state.texture_metadata[cat_idx]['category']
                                        # Calculer la couleur moyenne de la texture correspondante
                                        cat_images = st.session_state.texture_metadata[cat_idx]['images']
                                        all_pixels = []
                                        for img in cat_images:
                                            img_np = np.array(img) / 255.0
                                            all_pixels.append(img_np.reshape(-1, 3))
                                        if all_pixels:
                                            avg_texture_color = np.mean(np.vstack(all_pixels), axis=0)
                                            # Fusion réaliste : 70% couleur originale + 30% texture
                                            new_color = 0.7 * center_rgb + 0.3 * avg_texture_color
                                            # Appliquer au cluster
                                            mask = cluster_labels == c_id
                                            enhanced_colors[mask] = new_color
                                merged_colors = enhanced_colors
                                st.success("Textures PBR appliquées dynamiquement via correspondances FAISS (détection par similarité de couleurs/clusters).")
                   
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
                        
                        # Maillage si demandé (optimisé pour réalisme)
                        if generate_mesh:
                            try:
                                st.info("🔓 Générant et ouvrant fenêtre pour le maillage 3D ultra-réaliste...")
                                
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
                                
                                st.info("💡 Pour un rendu encore plus réaliste, exporte le maillage vers Blender/Unreal Engine en utilisant `mesh.export('mesh.ply')`.")
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
pip install streamlit plotly pillow numpy torch torchvision open3d scikit-learn transformers faiss-cpu
        """)
    st.markdown("**Lancer l'app :** `streamlit run app.py`")
    if st.button("🔗 Lien GitHub DUSt3R"):
        st.markdown("[https://github.com/naver/dust3r](https://github.com/naver/dust3r)")