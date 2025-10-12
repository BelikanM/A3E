# Import python-dotenv to load .env file
from dotenv import load_dotenv
import os
import shutil
import json
from pathlib import Path
import torch
import streamlit as st
from huggingface_hub import snapshot_download, HfApi
import requests
import queue
import threading
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    st.warning("psutil non installé. Installez-le avec 'pip install psutil python-dotenv' pour les diagnostics système et la gestion de l'environnement.")

try:
    from safetensors.torch import load_file
    SAFE_TENSORS_AVAILABLE = True
except ImportError:
    SAFE_TENSORS_AVAILABLE = False
    st.warning("safetensors non installé. Installez-le avec 'pip install safetensors' pour charger les fichiers .safetensors.")

# Load environment variables from .env file
load_dotenv()

# ===============================
# 🔑 Clé Hugging Face depuis variable d'environnement
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("⚠️ Hugging Face token non trouvé. Assurez-vous que le fichier .env existe dans le dossier racine avec la variable HF_TOKEN définie.")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ===============================
# CONFIGURATION DES CHEMINS
# ===============================
CACHE_PHOTOGRAM = Path(os.path.expanduser("~")) / ".cache/photogram_models"
CACHE_HF = Path(os.path.expanduser("~")) / ".cache/huggingface/hub"

# ===============================
# MODÈLES CUSTOM REQUÉRANT AJOUT À SYS.PATH
# ===============================
CUSTOM_REPOS = {"chenttt/matrix3d"}  # Ajoutez d'autres repos custom ici si nécessaire
CUSTOM_IMPORTS = {
    "chenttt/matrix3d": {
        "imports": [
            "from modeling import LRMGenerator, LRMGeneratorConfig",
            "from processor import LRMImageProcessor"
        ],
        "init_code": """
# Exemple d'initialisation
config = LRMGeneratorConfig()
model = LRMGenerator(config)
processor = LRMImageProcessor()
"""
    }
    # Ajoutez d'autres comme "facebook/vfusion3d" avec leurs imports spécifiques si disponibles
}

# ===============================
# LISTE DES MODÈLES PRÉDÉFINIS (STANDARDS)
# ===============================
LLM_MODELS = {
    "Llama 3.1 13B": "meta-llama/Meta-Llama-3.1-13B-Instruct",
    "Qwen 2.5 14B": "Qwen/Qwen2.5-14B-Instruct",
    "Gemma 2B": "google/gemma-2-2b-it",
    "SmolLM 3B": "HuggingFaceTB/SmolLM3-3B",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.2"
}
VISION_MODELS = {
    "DINOv2": "facebook/dinov2-base",
    "CLIP ViT-L/14": "openai/clip-vit-large-patch14",
    "SegFormer B5": "nvidia/segformer-b5-finetuned-ade-640-640"
}
VIDEO_MODELS = {
    "VideoMAE": "MCG-NJU/videomae-base",
    "Video-LDM": "CompVis/stable-diffusion-video"
}
SCIENTIFIC_MODELS = {
    "BioGPT": "microsoft/BioGPT-Large-PubMedQA",
    "SciBERT": "allenai/scibert_scivocab_uncased"
}
ALL_MODELS = {**LLM_MODELS, **VISION_MODELS, **VIDEO_MODELS, **SCIENTIFIC_MODELS}

# ===============================
# FONCTIONS UTILITAIRES
# ===============================
def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def get_model_size(path: Path):
    total = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass  # Ignore permission errors or other issues
    fmt = sizeof_fmt(total)
    return total, fmt

def get_model_structure(model_path: Path):
    """Retourne la structure relative du modèle : sous-dossiers et contenu (fichiers avec tailles)"""
    structure = []
    try:
        for root, dirs, files in os.walk(model_path):
            rel_root = Path(root).relative_to(model_path)
            if rel_root != Path('.'):
                structure.append({
                    'path': str(rel_root),
                    'type': 'dossier',
                    'content': f"{len(dirs)} sous-dossiers, {len(files)} fichiers"
                })
            for file in files:
                rel_file = Path(root) / file
                rel_path = rel_file.relative_to(model_path)
                size = rel_file.stat().st_size
                structure.append({
                    'path': str(rel_path),
                    'type': 'fichier',
                    'size': sizeof_fmt(size),
                    'content': f"Taille: {sizeof_fmt(size)}"
                })
    except Exception as e:
        structure.append({'path': 'Erreur', 'type': 'erreur', 'content': str(e)})
    return structure

def detect_cuda():
    return torch.cuda.is_available()

def get_system_diagnostics():
    diagnostics = {}
    if not PSUTIL_AVAILABLE:
        diagnostics["Erreur"] = "psutil requis pour les diagnostics."
        return diagnostics
    
    # CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    diagnostics["CPU"] = {
        "Cœurs physiques": cpu_count,
        "Threads logiques": cpu_threads,
        "Fréquence max (GHz)": f"{cpu_freq.max / 1000:.1f}" if cpu_freq else "N/A"
    }
    
    # RAM
    ram = psutil.virtual_memory()
    diagnostics["RAM"] = {
        "Totale (GB)": sizeof_fmt(ram.total, 'GB'),
        "Disponible (GB)": sizeof_fmt(ram.available, 'GB'),
        "Utilisation (%)": f"{ram.percent:.1f}%"
    }
    
    # CUDA/GPU
    if detect_cuda():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        diagnostics["GPU/CUDA"] = {
            "Disponible": "Oui",
            "Nom GPU": props.name,
            "Mémoire totale (GB)": sizeof_fmt(gpu_mem, 'GB'),
            "Utilisation mémoire (%)": f"{torch.cuda.memory_allocated(device) / gpu_mem * 100:.1f}%" if gpu_mem > 0 else "0%"
        }
    else:
        diagnostics["GPU/CUDA"] = {"Disponible": "Non"}
    
    # Conseils
    ram_gb = ram.total / (1024**3)
    if ram_gb < 8:
        diagnostics["Conseil RAM"] = "⚠️ RAM faible (<8GB). Limitez-vous aux petits modèles."
    elif ram_gb < 16:
        diagnostics["Conseil RAM"] = "ℹ️ RAM modérée (8-16GB). Convient pour modèles moyens."
    else:
        diagnostics["Conseil RAM"] = "✅ RAM suffisante (>16GB)."
    
    if detect_cuda():
        gpu_gb = gpu_mem / (1024**3)
        if gpu_gb < 4:
            diagnostics["Conseil GPU"] = "⚠️ VRAM faible (<4GB). Utilisez CPU ou petits modèles."
        elif gpu_gb < 8:
            diagnostics["Conseil GPU"] = "ℹ️ VRAM modérée (4-8GB). Bon pour modèles moyens."
        else:
            diagnostics["Conseil GPU"] = "✅ VRAM suffisante (>8GB)."
    else:
        diagnostics["Conseil GPU"] = "ℹ️ Pas de GPU CUDA. Utilisez CPU (plus lent)."
    
    return diagnostics

def detect_weights(path: Path):
    pt, tf = False, False
    try:
        for f in path.rglob('*'):
            if f.suffix in [".bin", ".safetensors"]:
                pt = True
            if f.suffix in [".h5"]:
                tf = True
    except Exception:
        pass  # Ignore permission errors
    return "Oui" if pt else "Non", "Oui" if tf else "Non"

def is_standard_model(model_dir: Path):
    """Détecte de manière robuste si le modèle est standard HF : config.json valide avec 'architectures', et absence de .py custom en racine"""
    try:
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            return False
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        has_arch = 'architectures' in config and isinstance(config['architectures'], list) and len(config['architectures']) > 0
        py_files = [f for f in model_dir.iterdir() if f.suffix == '.py' and f.name not in ['requirements.txt', 'setup.py']]
        no_custom_py = len(py_files) == 0
        return has_arch and no_custom_py
    except Exception:
        return False  # En cas d'erreur, considérer comme non-standard par prudence

def is_model_downloaded(repo_id, min_size=1024*1024):
    model_name = repo_id.replace("/", "--")
    path = CACHE_HF / f"models--{model_name}"
    if not path.exists():
        return False
    total, _ = get_model_size(path)
    return total >= min_size

def repo_exists(repo_id):
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        resp = requests.get(url, headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

def search_hf_models(query, limit=5):
    """Recherche des modèles sur Hugging Face et retourne la liste triée par popularité, avec détection robuste standard/custom"""
    try:
        api = HfApi(token=HF_TOKEN)
        model_infos = api.list_models(search=query, sort="downloads", direction=-1, limit=limit)
        results = []
        for info in model_infos:
            is_standard = False
            try:
                files = api.list_repo_files(info.id, repo_type="model", token=HF_TOKEN)
                root_py_files = [f for f in files if f.endswith('.py') and '/' not in f and f not in ['requirements.txt', 'setup.py']]
                try:
                    url = f"https://huggingface.co/{info.id}/resolve/main/config.json"
                    resp = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
                    if resp.status_code == 200:
                        config = json.loads(resp.text)
                        has_arch = 'architectures' in config and isinstance(config['architectures'], list) and len(config['architectures']) > 0
                        is_standard = has_arch and len(root_py_files) == 0
                except Exception:
                    pass  # Garde False si échec
            except Exception:
                is_standard = False
            results.append({
                'id': info.id,
                'author': info.author,
                'tags': info.tags,
                'downloads': info.downloads,
                'is_standard': is_standard
            })
        return results
    except Exception as e:
        st.error(f"Erreur lors de la recherche: {e}")
        return []

def download_worker(repo_id, result_queue):
    try:
        api = HfApi(token=HF_TOKEN)
        path = snapshot_download(repo_id, cache_dir=str(CACHE_HF), token=HF_TOKEN)
        result_queue.put(path)
    except Exception as e:
        result_queue.put(e)

def download_model(repo_id, auto_search=True, progress_bar=None):
    """Télécharge un modèle HuggingFace avec recherche automatique si le repo n'existe pas"""
    if is_model_downloaded(repo_id):
        return f"⚡ Le modèle '{repo_id}' est déjà téléchargé."
    if not repo_exists(repo_id):
        if auto_search:
            st.warning(f"❌ Le modèle '{repo_id}' n'existe pas. Recherche de modèles similaires...")
            query_term = repo_id.split("/")[-1]
            results = search_hf_models(query_term, limit=1)
            if results:
                best_match = results[0]['id']
                st.info(f"➡ Utilisation automatique du modèle trouvé : {best_match}")
                repo_id = best_match
            else:
                return f"❌ Aucun modèle correspondant trouvé pour '{repo_id}'."
        else:
            return f"❌ Le modèle '{repo_id}' n'existe pas ou n'est pas accessible."
    api = HfApi(token=HF_TOKEN)
    total_size = None
    try:
        repo_info = api.repo_info(repo_id=repo_id, token=HF_TOKEN)
        total_size = sum(s.size or 0 for s in repo_info.siblings)
    except Exception as e:
        st.warning(f"Impossible de récupérer la taille totale du repo: {e}. Téléchargement sans barre de progression.")
    model_name = repo_id.replace("/", "--")
    model_cache = CACHE_HF / f"models--{model_name}"
    result_queue = queue.Queue()
    thread = threading.Thread(target=download_worker, args=(repo_id, result_queue))
    thread.start()
    if progress_bar is not None:
        initial_size = 0
        if model_cache.exists():
            initial_size, _ = get_model_size(model_cache)
        progress_text = "Starting download..."
        progress_bar.progress(0, text=progress_text)
        while thread.is_alive():
            time.sleep(0.5)
            if model_cache.exists():
                current_size, _ = get_model_size(model_cache)
                downloaded = current_size - initial_size
                if total_size and total_size > 0:
                    progress = min(1.0, downloaded / total_size)
                    progress_text = f"Downloaded {sizeof_fmt(downloaded)} / {sizeof_fmt(total_size)}"
                else:
                    progress = 0
                    progress_text = f"Downloading... {sizeof_fmt(downloaded)}"
                progress_bar.progress(progress, text=progress_text)
        thread.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        else:
            path = result
            if progress_bar is not None:
                progress_bar.progress(1.0, text="Download complete!")
            return f"✅ Modèle '{repo_id}' téléchargé dans {path}"
    else:
        thread.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        else:
            path = result
            return f"✅ Modèle '{repo_id}' téléchargé dans {path}"

def cleanup_incomplete_models(threshold_bytes=1024*1024):
    """Supprime automatiquement les dossiers de modèles incomplets (taille < threshold)"""
    deleted = []
    for cache_root in [CACHE_PHOTOGRAM, CACHE_HF]:
        if cache_root.exists():
            try:
                for model_dir in list(cache_root.iterdir()):
                    if model_dir.is_dir():
                        total, _ = get_model_size(model_dir)
                        if total < threshold_bytes:
                            shutil.rmtree(model_dir)
                            deleted.append(str(model_dir))
            except Exception as e:
                st.error(f"Erreur lors du nettoyage de {cache_root}: {e}")
    return deleted

def cleanup_non_standard_models():
    """Supprime automatiquement les modèles non-standard (trop compliqués)"""
    deleted = []
    models = list_models()
    for model in models:
        if model['Standard HF'] == "Non":
            try:
                shutil.rmtree(model['Chemin'])
                deleted.append(model['Nom'])
            except Exception as e:
                st.error(f"Erreur suppression {model['Nom']}: {e}")
    return deleted

def list_models():
    models_list = []
    MIN_SIZE = 1024 * 1024  # 1MB
    for cache_root, model_type in [(CACHE_PHOTOGRAM, "Photogram"), (CACHE_HF, "HuggingFace")]:
        if not cache_root.exists():
            continue
        try:
            for model_dir in cache_root.iterdir():
                if model_dir.is_dir():
                    if model_type == "HuggingFace" and model_dir.name.startswith("models--"):
                        repo_slug = model_dir.name.replace("models--", "").replace("--", "/")
                    else:
                        repo_slug = model_dir.name
                    is_standard = is_standard_model(model_dir)
                    pt, tf = detect_weights(model_dir)
                    size_bytes, size = get_model_size(model_dir)
                    status = "Incomplet" if size_bytes < MIN_SIZE else "Complet"
                    models_list.append({
                        "Nom": model_dir.name,
                        "Chemin": str(model_dir),
                        "Taille": size,
                        "Statut": status,
                        "Type": model_type,
                        "PyTorch": pt,
                        "TF": tf,
                        "CUDA": "Oui" if detect_cuda() else "Non",
                        "Standard HF": "Oui" if is_standard else "Non",
                        "RepoID": repo_slug
                    })
        except Exception:
            pass  # Ignore errors in listing
    return models_list

def find_safetensors_files(model_path: Path):
    """
    Trouve les fichiers .safetensors dans un dossier de modèle HF (avec snapshots ou en racine).
    """
    safetensors_files = []
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        for snapshot_dir in snapshots_dir.iterdir():
            for f in snapshot_dir.glob("*.safetensors"):
                safetensors_files.append(f)
    else:
        # Si pas de snapshots, cherche en racine
        for f in model_path.glob("*.safetensors"):
            safetensors_files.append(f)
    return safetensors_files

def load_safetensors_weights(file_path: Path):
    """
    Charge les poids d'un fichier .safetensors de manière sécurisée.
    """
    if not SAFE_TENSORS_AVAILABLE:
        raise ImportError("safetensors requis pour charger les fichiers .safetensors. Installez-le avec 'pip install safetensors'.")
    device = "cpu"
    return load_file(str(file_path), device=device)

# ===============================
# INTERFACE STREAMLIT
# ===============================
st.set_page_config(page_title="Gestionnaire de Modèles", layout="wide")
st.title("📦 Gestionnaire de modèles")

# -------------------------------
# Sidebar pour diagnostics
# -------------------------------
st.sidebar.header("🔧 Diagnostics Système")
diagnostics = get_system_diagnostics()
for section, info in diagnostics.items():
    if isinstance(info, dict):
        with st.sidebar.expander(f"{section}"):
            for key, value in info.items():
                st.sidebar.write(f"**{key}:** {value}")
    else:
        st.sidebar.write(f"**{section}:** {info}")

# -------------------------------
# Modèles déjà en cache
# -------------------------------
st.subheader("📂 Modèles dans le cache")
models = list_models()
if models:
    df = st.dataframe(
        models,
        use_container_width=True,
        column_config={
            "Chemin": st.column_config.TextColumn("Chemin", help="Cliquez sur 'Détails' pour voir la structure"),
            "Nom": st.column_config.TextColumn("Nom", help="Nom du modèle"),
            "Standard HF": st.column_config.TextColumn("Standard HF", help="Oui: Compatible AutoModel, plug-and-play. Non: Custom, nécessite code spécifique, non recommandé.")
        }
    )
    for model in models:
        with st.expander(f"🔍 Détails pour {model['Nom']} (survol/copie chemins relatifs)"):
            model_path = Path(model['Chemin'])
            structure = get_model_structure(model_path)
            if structure:
                st.write("**Structure relative du modèle :**")
                for item in structure:
                    if item['type'] == 'dossier':
                        st.write(f"📁 **{item['path']}** - {item['content']}")
                    elif item['type'] == 'fichier':
                        st.code(item['path'], language='text')
                        st.caption(item['content'])
                    else:
                        st.error(item['content'])
            
            if model['Standard HF'] == "Non":
                st.warning("⚠️ Modèle non-standard: Chargement complexe, peut nécessiter trust_remote_code=True et imports manuels. Non recommandé.")
            
            st.write("**Code d'exemple pour charger le modèle :**")
            repo_id = model['RepoID']
            if model['Standard HF'] == "Oui":
                load_code = f"""from transformers import AutoModel

model = AutoModel.from_pretrained(
    r"{model['Chemin']}", 
    trust_remote_code=False
)
"""
                st.code(load_code, language="python")
            else:
                if repo_id in CUSTOM_IMPORTS:
                    imports_list = CUSTOM_IMPORTS[repo_id]["imports"]
                    init_code = CUSTOM_IMPORTS[repo_id]["init_code"]
                    load_code = f"""import sys
from pathlib import Path

MODEL_PATH = Path(r"{model['Chemin']}")
sys.path.append(str(MODEL_PATH))

{chr(10).join(imports_list)}

{init_code}"""
                    st.code(load_code, language="python")
                else:
                    load_code = f"""import sys
from pathlib import Path

MODEL_PATH = Path(r"{model['Chemin']}")
sys.path.append(str(MODEL_PATH))

# Importer les modules custom (adaptez selon le modèle)
# from modeling import YourModelClass
# from processor import YourProcessorClass"""
                    st.code(load_code, language="python")
            
            st.info("**Utilisation exemple :** Copiez ce code dans votre script Python pour charger le modèle.")

            # Nouvelle fonctionnalité : Chargement direct des poids pour modèles non-standard HF
            if model['Type'] == "HuggingFace" and model['Standard HF'] == "Non":
                st.subheader("🔧 Chargement direct des poids (.safetensors)")
                safetensors = find_safetensors_files(model_path)
                if safetensors:
                    st.write("**Fichiers .safetensors disponibles :**")
                    for f in safetensors:
                        rel_path = f.relative_to(model_path)
                        st.write(f"• `{rel_path}` (taille: {sizeof_fmt(f.stat().st_size)})")

                    # Sélectionne le plus grand fichier (souvent le principal)
                    main_file = max(safetensors, key=lambda p: p.stat().st_size)
                    device_str = "cpu"
                    load_weights_code = f"""from safetensors.torch import load_file
from pathlib import Path

file_path = Path(r\"{main_file}\")
state_dict = load_file(str(file_path), device=\"{device_str}\")

# Aperçu
if state_dict:
    first_key = next(iter(state_dict))
    print(f\"Clé: {{first_key}}\")
    print(f\"Shape: {{state_dict[first_key].shape}}\")
"""
                    st.write("**Exemple de code pour charger les poids :**")
                    st.code(load_weights_code, language="python")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("🔍 Aperçu des poids", key=f"preview_weights_{model['Nom']}"):
                            with st.spinner("Chargement des poids... (peut prendre du temps pour gros modèles)"):
                                try:
                                    if not SAFE_TENSORS_AVAILABLE:
                                        raise ImportError("safetensors requis. Installez-le avec 'pip install safetensors'.")
                                    state_dict = load_safetensors_weights(main_file)
                                    st.success("✅ Poids chargés avec succès !")
                                    if state_dict:
                                        first_key = next(iter(state_dict))
                                        shape = state_dict[first_key].shape
                                        num_keys = len(state_dict)
                                        st.info(f"**Première clé :** `{first_key}`")
                                        st.info(f"**Shape du tensor :** `{shape}`")
                                        st.write(f"**Nombre total de clés :** {num_keys:,}")
                                    else:
                                        st.warning("State dict vide.")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du chargement : {str(e)}")
                                    st.caption("Vérifiez si safetensors est installé ou essayez manuellement.")
                else:
                    st.warning("❌ Aucun fichier .safetensors trouvé dans ce modèle.")
            
else:
    st.info("Aucun modèle trouvé dans les caches.")

# -------------------------------
# Nettoyage automatique
# -------------------------------
st.subheader("🧹 Nettoyage des modèles incomplets")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Nettoyer automatiquement les modèles incomplets (<1MB)"):
        with st.spinner("Nettoyage en cours..."):
            deleted = cleanup_incomplete_models()
        if deleted:
            st.success(f"✅ {len(deleted)} modèles incomplets supprimés.")
            for d in deleted:
                st.write(f"- {d}")
        else:
            st.info("ℹ️ Aucun modèle incomplet trouvé.")
        models = list_models()
        st.dataframe(models, use_container_width=True)
with col2:
    threshold = st.number_input("Seuil de taille (octets)", value=1024*1024, min_value=0)
    if st.button("Nettoyer avec seuil personnalisé"):
        with st.spinner("Nettoyage en cours..."):
            deleted = cleanup_incomplete_models(threshold)
        if deleted:
            st.success(f"✅ {len(deleted)} modèles supprimés.")
            for d in deleted:
                st.write(f"- {d}")
        else:
            st.info("ℹ️ Aucun modèle à supprimer.")
        models = list_models()
        st.dataframe(models, use_container_width=True)
with col3:
    if st.button("Supprimer les modèles non-standard (trop compliqués)"):
        with st.spinner("Suppression des non-standards en cours..."):
            deleted = cleanup_non_standard_models()
        if deleted:
            st.success(f"✅ {len(deleted)} modèles non-standard supprimés.")
            for d in deleted:
                st.write(f"- {d}")
        else:
            st.info("ℹ️ Aucun modèle non-standard trouvé.")
        models = list_models()
        st.dataframe(models, use_container_width=True)

# -------------------------------
# Téléchargement modèle prédéfini
# -------------------------------
st.subheader("⬇ Télécharger un modèle prédéfini")
model_names = list(ALL_MODELS.keys())
selected_model = st.selectbox("Choisir un modèle", ["-- Sélectionner --"] + model_names)
download_btn = st.button("Télécharger le modèle")
if download_btn and selected_model != "-- Sélectionner --":
    repo_id = ALL_MODELS[selected_model]
    progress_bar = st.progress(0, text=f"Starting download of {selected_model}...")
    with st.spinner(f"Téléchargement de {selected_model}..."):
        try:
            status = download_model(repo_id, auto_search=True, progress_bar=progress_bar)
        except Exception as e:
            status = f"❌ Erreur téléchargement: {e}"
            if progress_bar:
                progress_bar.progress(0, text="Download failed")
        st.success(status)
        models = list_models()
        st.dataframe(models, use_container_width=True)

# -------------------------------
# Recherche Hugging Face live
# -------------------------------
st.subheader("🔍 Rechercher un modèle Hugging Face")
col_search, _ = st.columns([1, 3])
with col_search:
    show_all = st.checkbox("Montrer tous (incl. non-standard)", value=False)
query = st.text_input("Mot-clé de recherche (ex: llama, clip, bio)")
if query:
    with st.spinner("Recherche en cours..."):
        results = search_hf_models(query, limit=10)
    filtered_results = results if show_all else [r for r in results if r['is_standard']]
    if filtered_results:
        for model in filtered_results:
            tags_str = ', '.join(model['tags'][:5]) if model['tags'] else 'Aucun'
            label = " (Standard HF - Recommandé)" if model['is_standard'] else " (Non-standard - Non recommandé)"
            st.write(f"**{model['id']}{label}** - {model['author'] or 'Inconnu'} - Tags: {tags_str} - Downloads: {model['downloads']}")
            if st.button(f"Télécharger {model['id']}", key=f"dl_{model['id']}"):
                progress_bar = st.progress(0, text=f"Starting download of {model['id']}...")
                with st.spinner(f"Téléchargement de {model['id']}..."):
                    try:
                        status = download_model(model['id'], auto_search=False, progress_bar=progress_bar)
                    except Exception as e:
                        status = f"❌ Erreur téléchargement: {e}"
                        if progress_bar:
                            progress_bar.progress(0, text="Download failed")
                    st.success(status)
                    models = list_models()
                    st.dataframe(models, use_container_width=True)
    else:
        st.warning("Aucun modèle trouvé pour cette recherche (essayez 'Montrer tous').")

# -------------------------------
# Rafraîchir manuellement
# -------------------------------
if st.button("🔄 Rafraîchir la liste des modèles"):
    models = list_models()
    st.dataframe(models, use_container_width=True)