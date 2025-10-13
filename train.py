# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io
import plotly.graph_objects as go
import os

# ----------------------------
# Model: Multiview -> per-view splats (x,y,z,sigma)
# ----------------------------
class MultiviewSplatModel(nn.Module):
    def __init__(self, feature_dim=512, use_pretrained=False):
        super().__init__()
        self.use_pretrained = use_pretrained
        if use_pretrained:
            try:
                from torchvision.models import resnet18, ResNet18_Weights
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # output [B,512,1,1]
                out_dim = 512
            except Exception:
                # fallback small conv
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                out_dim = 64
        else:
            # small conv encoder (no internet)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            out_dim = 256

        # map each view feature to 4 dims (x,y,z,sigma)
        self.project = nn.Linear(out_dim, 128)
        self.head = nn.Linear(128, 4)

    def forward(self, images):  # images: [views, 3, H, W]
        v = images.shape[0]
        x = self.encoder(images)  # [v, C, 1,1]
        x = x.view(v, -1)         # [v, C]
        x = F.relu(self.project(x))
        splats = self.head(x)     # [v, 4]
        # apply small normalization for coords/sigma
        coords = torch.tanh(splats[:, :3])  # x,y,z in [-1,1]
        sigma = F.softplus(splats[:, 3])    # >0
        return torch.cat([coords, sigma.unsqueeze(1)], dim=1)  # [v,4]


# ----------------------------
# Utilities
# ----------------------------
def read_image_file(file, size=(256,256)):
    img = Image.open(file).convert("RGB").resize(size)
    return img

preprocess = T.Compose([
    T.ToTensor(),                 # [C,H,W] float32 0..1
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def group_files_by_prefix(files):
    # Expect filenames like obj1_view1.jpg -> prefix before first '_'
    groups = {}
    for f in files:
        name = os.path.basename(f.name)
        prefix = name.split('_')[0] if '_' in name else 'object'
        groups.setdefault(prefix, []).append(f)
    return groups

def write_ply(points, filename):
    # points: N x [x,y,z, sigma]
    N = points.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property float radius
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Multiview Gaussian Splatting")
st.title("Multiview Gaussian Splatting — Demo")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Chargement & paramètres")
    uploaded = st.file_uploader(
        "Upload images (naming: objectA_view1.jpg, objectA_view2.jpg, objectB_view1.jpg...)",
        accept_multiple_files=True,
        type=['png','jpg','jpeg']
    )
    use_pretrained = st.checkbox("Utiliser encoder pré-entrainé (ResNet18)", value=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write("Device:", device)
    size = st.selectbox("Taille des images (resize)", options=[128, 256, 384], index=1)
    splat_scale = st.slider("Échelle taille des splats (affichage)", 1.0, 50.0, 10.0)
    sigma_scale = st.slider("Échelle sigma (export PLY radius)", 0.01, 1.0, 0.1)
    cmap = st.selectbox("Colormap (Plotly)", ["Viridis","Hot","Cividis","Plasma","Inferno"], index=0)

    if st.button("Recharger modèle"):
        st.experimental_rerun()

with col2:
    st.header("Visualisation 3D")
    fig_placeholder = st.empty()

# ----------------------------
# Processing
# ----------------------------
if uploaded and len(uploaded) >= 2:
    groups = group_files_by_prefix(uploaded)
    st.write("Groupes détectés :", list(groups.keys()))
    model = MultiviewSplatModel(use_pretrained=use_pretrained).to(device)
    model.eval()

    all_objects_results = {}
    for obj_name, files in groups.items():
        # sort by filename for deterministic order
        files = sorted(files, key=lambda f: f.name)
        imgs = []
        for f in files:
            img = read_image_file(f, size=(size, size))
            st.image(img, width=100, caption=f"{obj_name} / {f.name}")
            t = preprocess(img).to(device)
            imgs.append(t)
        imgs_tensor = torch.stack(imgs, dim=0)  # [views, 3, H, W]
        with torch.no_grad():
            splats = model(imgs_tensor)  # [views,4]
        splats_np = splats.cpu().numpy()
        all_objects_results[obj_name] = {
            'files': [f.name for f in files],
            'splats': splats_np
        }

    # Build Plotly figure with all objects
    fig = go.Figure()
    colorbar_shown = False
    for obj_name, info in all_objects_results.items():
        s = info['splats']
        x,y,z,sig = s[:,0], s[:,1], s[:,2], s[:,3]
        sizes = (sig * splat_scale).clip(min=1.0)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            text=[obj_name]*len(x),
            marker=dict(size=sizes, color=sig, colorscale=cmap, showscale=not colorbar_shown, opacity=0.9),
            name=obj_name
        ))
        colorbar_shown = True

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      margin=dict(l=0,r=0,b=0,t=0), height=700, width=900)
    fig_placeholder.plotly_chart(fig, use_container_width=True)

    # Show tables & export
    st.header("Résultats détaillés")
    for obj_name, info in all_objects_results.items():
        st.subheader(obj_name)
        st.table(info['splats'])
        if st.button(f"Exporter PLY pour {obj_name}"):
            # Build ply scaled by sigma_scale
            points = info['splats'].copy()
            points[:,3] = points[:,3] * sigma_scale
            out_name = f"{obj_name}_splats.ply"
            write_ply(points, out_name)
            with open(out_name, "rb") as fh:
                data = fh.read()
            st.download_button(label=f"Télécharger {out_name}", data=data, file_name=out_name, mime="application/octet-stream")

else:
    st.info("Téléverse au moins 2 images (de préférence groupées par objet). Voir la colonne de gauche pour options.")
