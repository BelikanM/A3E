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

# Imports spécifiques à DUSt3R (assurez-vous d'avoir installé : pip install git+https://github.com/naver/dust3r.git)
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as dust3r_load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import xy_grid

# Imports pour MapAnything (assurez-vous d'avoir installé : git clone https://github.com/facebookresearch/map-anything.git && cd map-anything && pip install -e .)
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
def load_map_model():
    try:
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        st.success("Modèle MapAnything chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle MapAnything : {e}")
        st.info("Assurez-vous d'avoir installé MapAnything : `git clone https://github.com/facebookresearch/map-anything.git && cd map-anything && pip install -e .`")
        return None

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
    model_choice = st.radio("Modèle de reconstruction", ["DUSt3R", "MapAnything"], help="Choisissez DUSt3R pour une approche stéréo ou MapAnything pour une reconstruction universelle metric 3D.")
    
    if model_choice == "DUSt3R":
        batch_size = st.slider("Taille du batch", min_value=1, max_value=4, value=1, help="Nombre d'images traitées simultanément (plus petit = plus stable sur GPU)")
        niter_align = st.slider("Itérations d'alignement global", min_value=100, max_value=500, value=300, help="Nombre d'itérations pour l'optimisation globale")
        lr_align = st.slider("Taux d'apprentissage alignement", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    else:
        # Options spécifiques à MapAnything
        use_amp = st.checkbox("Utiliser AMP (bf16)", value=True, help="Accélération mixte de précision pour CUDA")
        confidence_percentile = st.slider("Percentile de confiance", min_value=0, max_value=50, value=10, help="Filtre de confiance pour les points (plus élevé = plus strict)")
    
    threshold_conf = st.slider("Seuil de confiance", min_value=0.0, max_value=1.0, value=0.5, format="%.2f", help="Seuil pour filtrer les points de confiance")
    max_points_per_view = st.slider("Max points par vue (downsample)", min_value=1000, max_value=100000, value=20000, help="Nombre max de points par image pour visualisation HD")
    generate_mesh = st.checkbox("Générer maillage 3D propre", value=False, help="Crée un maillage complet à partir du nuage de points avec textures ultra-réalistes.")
    poisson_depth = st.slider("Profondeur maillage (Poisson)", min_value=5, max_value=12, value=9, help="Niveau de détail pour la reconstruction Poisson (plus élevé = plus fin).")
   
    process_btn = st.button("🚀 Lancer la Reconstruction 3D", type="primary")

with col2:
    if uploaded_files and len(uploaded_files) >= 2 and process_btn:
        model = load_dust3r_model() if model_choice == "DUSt3R" else load_map_model()
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
                           
                            # Mode conditionnel basé sur le nombre d'images
                            num_images = len(images)
                            mode = GlobalAlignerMode.PointCloudOptimizer if num_images > 2 else GlobalAlignerMode.PairViewer
                            scene = global_aligner(
                                output,
                                device=device,
                                mode=mode
                            )
                           
                            loss_value = 0.0
                            if mode == GlobalAlignerMode.PointCloudOptimizer:
                                loss = scene.compute_global_alignment(
                                    init="mst",
                                    niter=niter_align,
                                    schedule='cosine',
                                    lr=lr_align
                                )
                                loss_value = loss
                                progress_bar.progress(1.0)
                                status_text.text(f"Alignement terminé ! Perte finale : {loss:.4f}")
                            else:
                                progress_bar.progress(1.0)
                                status_text.text("Vue paire traitée ! (Pas d'alignement global pour 2 images)")
                           
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
                       
                        else:  # MapAnything
                            # Chargement manuel des vues pour MapAnything (inspiré de DUSt3R : redimensionnement à 512x512 et normalisation)
                            status_text.text("Chargement des images MapAnything...")
                            views = []
                            for path in img_paths:
                                pil_img = Image.open(path).convert("RGB")
                                pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
                                img_array = np.array(pil_img).astype(np.float32) / 255.0
                                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
                                
                                views.append({
                                    "img": img_tensor,
                                    "data_norm_type": "imagenet"  # Clé obligatoire pour le modèle
                                })
                           
                            status_text.text("Inférence MapAnything en cours...")
                            predictions = model.infer(
                                views,
                                memory_efficient_inference=False,
                                use_amp=use_amp and device == 'cuda',
                                amp_dtype="bf16",
                                apply_mask=True,
                                mask_edges=True,
                                apply_confidence_mask=False,
                                confidence_percentile=confidence_percentile,
                            )
                           
                            progress_bar.progress(1.0)
                            status_text.text("Inférence MapAnything terminée ! (Note: Reconstructions indépendantes par vue, sans alignement global automatique)")
                           
                            # Récupération des résultats MapAnything (liste de dicts, un par vue)
                            all_pts3d = []
                            all_colors = []
                            H, W = 512, 512
                            for i, pred in enumerate(predictions):
                                # Assumer B=1 pour chaque vue
                                pts3d_i_full = pred["pts3d"][0].detach().cpu().numpy().reshape(-1, 3)  # (H*W, 3)
                                conf_i_full = pred["conf"][0].detach().cpu().numpy()  # (H, W)
                                conf_i_flat = conf_i_full.flatten()  # (H*W,)
                                
                                # Couleurs depuis l'image d'entrée normalisée (0-1)
                                img_tensor = views[i]["img"]  # (3, H, W)
                                colors_i_full = img_tensor.permute(1, 2, 0).detach().cpu().numpy().reshape(-1, 3)  # (H*W, 3)
                                
                                # Filtrage par confiance
                                valid_mask = conf_i_flat > threshold_conf
                                pts3d_i = pts3d_i_full[valid_mask]
                                colors_i = colors_i_full[valid_mask]
                                
                                if len(pts3d_i) == 0:
                                    st.warning(f"Aucun point de confiance pour l'image {i+1}")
                                    continue
                                
                                # Downsample
                                n_valid = len(pts3d_i)
                                if n_valid > max_points_per_view:
                                    down_idx = np.random.choice(n_valid, max_points_per_view, replace=False)
                                    pts3d_i = pts3d_i[down_idx]
                                    colors_i = colors_i[down_idx]
                                
                                all_pts3d.append(pts3d_i)
                                all_colors.append(colors_i)
                           
                            num_pairs = len(views) * (len(views) - 1) // 2
                            loss_value = 0.0  # Pas de perte pour MapAnything (feed-forward)
                   
                    # Fusion des nuages de points (après le with, mais arrays persistants)
                    if all_pts3d:
                        merged_pts3d = np.vstack(all_pts3d)
                        merged_colors = np.vstack(all_colors)
                    else:
                        merged_pts3d = np.empty((0, 3))
                        merged_colors = np.empty((0, 3))
                   
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
                            st.info("🔓 Générant et ouvrant fenêtre pour le maillage 3D ultra-réaliste...")
                            
                            # Downsampling intelligent avant Poisson (évite surcharge mémoire)
                            target_voxel_size = 0.005  # 5 mm pour un scan précis
                            pcd = pcd.voxel_down_sample(voxel_size=target_voxel_size)
                            
                            # Estimation plus robuste des normales
                            pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
                            )
                            pcd.orient_normals_consistent_tangent_plane(100)  # Rendre les normales cohérentes
                            
                            # Reconstruction Poisson avec nettoyage
                            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
                            )
                            
                            # Supprimer les triangles peu denses (artefacts)
                            vertices_to_remove = densities < np.quantile(densities, 0.01)
                            mesh.remove_vertices_by_mask(vertices_to_remove)
                            
                            # Couleurs réalistes sur le maillage (alignées après downsampling)
                            mesh.vertex_colors = pcd.colors
                            
                            # Visualisation avancée du maillage
                            o3d.visualization.draw_geometries(
                                [mesh],
                                window_name=f"Maillage 3D Poisson Ultra-Réaliste - {model_choice}",
                                width=1600,
                                height=900,
                                mesh_show_back_face=True,  # Montre les faces arrière
                                point_show_normal=False
                            )
                            
                            st.info("💡 Pour un rendu encore plus réaliste, exporte le maillage vers Blender/Unreal Engine en utilisant `mesh.export('mesh.ply')`.")
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
                                colorbar=dict(title="Profondeur (Z)")
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
    model_choice_placeholder = st.radio("Sélectionnez pour voir les instructions :", ["DUSt3R", "MapAnything"], key="install_choice")
    if model_choice_placeholder == "DUSt3R":
        st.code("""
pip install git+https://github.com/naver/dust3r.git
pip install streamlit plotly pillow numpy torch torchvision open3d
        """)
    else:
        st.code("""
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything
pip install -e .
pip install streamlit plotly pillow numpy torch torchvision open3d
        """)
    st.markdown("**Lancer l'app :** `streamlit run app.py`")
    if st.button("🔗 Lien GitHub DUSt3R"):
        st.markdown("[https://github.com/naver/dust3r](https://github.com/naver/dust3r)")
    if st.button("🔗 Lien GitHub MapAnything"):
        st.markdown("[https://github.com/facebookresearch/map-anything](https://github.com/facebookresearch/map-anything)")
