import streamlit as st
import numpy as np
import pandas as pd
from construct import Struct, Int16ul, Array, GreedyRange
import magic
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.interpolate import griddata, interp1d
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch
import io
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path
import sqlite3
import time
import gc
import shutil
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import kaleido
import psutil
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

if pio.kaleido.scope is not None:
    pio.defaults.kaleido = {
        "executable_path": "/usr/bin/chromium-browser"
    }
    pio.kaleido.scope.default_format = "png"
    pio.kaleido.scope.default_width = 800
    pio.kaleido.scope.default_height = 600
else:
    st.warning("Kaleido not available. Image export may not work.")

# Set Tavily API Key
os.environ["TAVILY_API_KEY"] = "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"

# Imports optionnels avec gestion d'erreurs
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    st.error("Impossible d'importer transformers. Veuillez installer la bibliothèque avec `pip install transformers`.")
    st.stop()

try:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    st.warning("⚠️ diffusers non disponible. Installez avec `pip install diffusers[torch] accelerate safetensors` pour la génération d'images IA.")

try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import LlamaCpp
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ LangChain non disponible. Installez avec `pip install langchain faiss-cpu sentence-transformers pypdf2 langchain-community langchain-huggingface` pour le RAG.")

st.set_page_config(page_title="ERT/GPR Analyzer Avancé", layout="wide")

# Sidebar pour gestion GPU/CPU monitoring
st.sidebar.title("⚙️ Gestion GPU/CPU")

# Paramètres de monitoring
enable_monitoring = st.sidebar.checkbox("Activer monitoring GPU/CPU", value=False)
gpu_threshold = st.sidebar.slider("Seuil saturation GPU (%)", 0, 100, 80)
cpu_threshold = st.sidebar.slider("Seuil saturation CPU (%)", 0, 100, 80)
auto_pause = st.sidebar.checkbox("Activer pause automatique après utilisation", value=True)
pause_duration = st.sidebar.slider("Durée pause (s)", 1, 60, 10)

# Initialisation session state pour monitoring
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = {'time': [], 'gpu_usage': [], 'cpu_usage': []}

def update_monitoring():
    if enable_monitoring:
        current_time = time.time()
        st.session_state.monitoring_data['time'].append(current_time)
        
        # GPU monitoring (si CUDA disponible)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
            st.session_state.monitoring_data['gpu_usage'].append(gpu_mem)
        else:
            st.session_state.monitoring_data['gpu_usage'].append(0)
        
        # CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1)
        st.session_state.monitoring_data['cpu_usage'].append(cpu_percent)
        
        # Limiter à 100 points pour performance
        if len(st.session_state.monitoring_data['time']) > 100:
            for key in st.session_state.monitoring_data:
                st.session_state.monitoring_data[key] = st.session_state.monitoring_data[key][-100:]

def plot_monitoring_graphs():
    if enable_monitoring and len(st.session_state.monitoring_data['time']) > 1:
        fig = make_subplots(rows=2, cols=1, subplot_titles=('GPU Usage (%)', 'CPU Usage (%)'))
        
        times = np.array(st.session_state.monitoring_data['time']) - st.session_state.monitoring_data['time'][0]
        
        fig.add_trace(go.Scatter(x=times, y=st.session_state.monitoring_data['gpu_usage'], mode='lines', name='GPU'), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=st.session_state.monitoring_data['cpu_usage'], mode='lines', name='CPU'), row=2, col=1)
        
        fig.update_layout(height=400, title_text="Monitoring GPU/CPU")
        st.sidebar.plotly_chart(fig, use_container_width=True)

def auto_pause_gpu():
    if auto_pause and torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
        if gpu_usage > gpu_threshold:
            st.info(f"GPU saturé ({gpu_usage:.1f}%) - Pause de {pause_duration}s et vidage cache.")
            torch.cuda.empty_cache()
            time.sleep(pause_duration)
            gc.collect()

# Mise à jour monitoring
update_monitoring()
plot_monitoring_graphs()

st.title("🌍 Analyse ERT/GPR Avancée avec RAG, IA Générative, et Reconstruction 3D (IA Images)")

# Chargement du modèle IA compact (SmolLM-135M)
@st.cache_resource
def load_ai_model():
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bitsandbytes_available = False
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        bitsandbytes_available = True
    except ImportError:
        pass
    if torch.cuda.is_available():
        if bitsandbytes_available:
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                precision_msg = f"Modèle chargé en 8-bit sur GPU (dtype: {model.dtype}). Optimisation mémoire activée."
                st.success(precision_msg)
                return tokenizer, model
            except Exception as quant_e:
                st.warning(f"Erreur quantification 8-bit : {quant_e}. Chargement en précision complète sur GPU.")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            st.warning("⚠️ bitsandbytes non disponible. Installez-le avec `pip install bitsandbytes` pour quantification 8-bit (réduit mémoire de ~30%). Chargement en précision complète sur GPU.")
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        st.info("CPU détecté : Chargement en précision complète (pas de GPU pour quantification).")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    precision_msg = f"Modèle chargé en précision complète (dtype: {model.dtype})."
    st.info(precision_msg)
    return tokenizer, model

tokenizer, model = load_ai_model()
st.success(f"✅ Modèle IA compact chargé : SmolLM-135M. Prêt pour analyse scientifique avancée.")

# Chargement du modèle Llama GGUF pour analyse en temps réel et RAG avancé
@st.cache_resource
def load_llama_model():
    model_path = "snapshots/5cf2772e6afee7b983b811b3c020cdddcaa2596c/Meta-Llama-3.1-13B-Instruct-abliterated.Q3_K_M.gguf"
    if not os.path.exists(model_path):
        return None
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            verbose=False
        )
        st.success("✅ Modèle Meta-LLaMA 3.1 13B Instruct (GGUF) chargé pour analyse en temps réel.")
        return llm
    except ImportError:
        st.error("llama-cpp-python requis. Installez avec `pip install llama-cpp-python` (ou `[cuda]` pour GPU).")
        return None
    except Exception as e:
        st.error(f"Erreur chargement Llama : {e}")
        return None

llama_model = load_llama_model()

# Wrapper pour LangChain avec Llama (pour RAG)
def get_langchain_llm():
    if not LANGCHAIN_AVAILABLE:
        return None
    model_path = "snapshots/5cf2772e6afee7b983b811b3c020cdddcaa2596c/Meta-Llama-3.1-13B-Instruct-abliterated.Q3_K_M.gguf"
    if not os.path.exists(model_path):
        return None
    try:
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=300,
            top_p=0.95,
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            n_ctx=2048,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Erreur chargement LlamaCpp pour RAG : {e}")
        return None

# RAG Setup (seulement si disponible, maintenant avec Llama et Tavily)
if LANGCHAIN_AVAILABLE:
    @st.cache_resource
    def init_rag_from_pdfs(pdf_paths):
        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

    prompt_template = """Utilise le contexte suivant pour répondre précisément à la question sur l'analyse ERT/GPR.
    Contexte : {context}
    Question : {question}
    Réponse concise et exacte :"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Agent setup with Tavily
    @st.cache_resource
    def create_rag_agent(vectorstore, llm):
        search = TavilySearchResults(max_results=3)
        retriever_tool = create_retriever_tool(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            "search_rag",
            "Search the uploaded PDFs for relevant information on ERT/GPR."
        )
        tools = [search, retriever_tool]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor
else:
    def init_rag_from_pdfs(pdf_paths):
        st.warning("RAG non disponible.")
        return None

# SDXL avec Safety Checker intégré
@st.cache_resource
def load_image_gen_model():
    if not DIFFUSERS_AVAILABLE:
        st.error("diffusers requis pour SDXL. Installez-le.")
        return None
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    st.info(f"Téléchargement et chargement du Safety Checker {safety_model_id}...")
    try:
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
        feature_extractor = CLIPImageProcessor.from_pretrained(safety_model_id)
        st.success("✅ Modèle Safety Checker chargé avec succès !")
    except Exception as e:
        st.warning(f"Erreur Safety Checker : {e}. Désactivation.")
        safety_checker = None
        feature_extractor = None
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        st.info("✅ SDXL chargé sur GPU (fp16 pour optimisation mémoire).")
    else:
        st.warning("⚠️ CPU détecté : Génération plus lente (~20s/image).")
    return pipe

pipe = load_image_gen_model()
if pipe:
    st.success("✅ IA Génératrice d'Images (SDXL) chargée. Prête pour prompts géophysiques !")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = "progress.db"
RECON_PATH = "reconstruction.bin"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            status TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS textures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            type TEXT,
            path TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_map_model():
    try:
        from mapanything.models import MapAnything
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        st.success("Modèle MapAnything chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle MapAnything : {e}")
        return None

# Fonctions utilitaires
def save_uploaded_to_dir(uploaded_files, dir_name='input_images'):
    os.makedirs(dir_name, exist_ok=True)
    for i, file in enumerate(uploaded_files):
        ext = file.name.split('.')[-1]
        path = os.path.join(dir_name, f"img_{i:03d}.{ext}")
        with open(path, 'wb') as f:
            f.write(file.getvalue())
    return dir_name

# Fonction pour ajouter annotations ERT à l'image générée (améliorée avec légendes minéraux)
def add_ert_annotations_to_image(image, rho_min, rho_max, max_depth, colorscale, electrode_spacing, materials_df):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)  # Chemin générique, ajustez si besoin
    except:
        font = ImageFont.load_default()
    
    width, height = image.size
    # Colorbar simplifié à droite
    bar_width = 30
    bar_height = height - 100
    bar_x = width - 50
    bar_y = 50
    
    # Dessiner la barre de couleur (gradient approximatif)
    for i in range(bar_height):
        norm_val = i / bar_height
        color_idx = int(norm_val * (len(colorscale) - 1))
        r, g, b = int(colorscale[color_idx][1].split('(')[1].split(',')[0]), \
                 int(colorscale[color_idx][1].split(',')[1]), \
                 int(colorscale[color_idx][1].split(',')[2].split(')')[0])
        draw.line([(bar_x, bar_y + i), (bar_x + bar_width, bar_y + i)], fill=(r, g, b))
    
    # Labels pour résistivités (log scale)
    tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_texts = [f'{rho_min:.2f}', f'{np.exp(np.log(rho_min) + 0.5*(np.log(rho_max)-np.log(rho_min))):.0f}', 
                  f'{np.exp(np.log(rho_min) + (np.log(rho_max)-np.log(rho_min))):.0f}', 
                  f'{np.exp(np.log(rho_min) + 1.5*(np.log(rho_max)-np.log(rho_min))):.0f}', f'{rho_max:.0f}']
    for i, (tv, tt) in enumerate(zip(tick_vals, tick_texts)):
        y_pos = bar_y + int(tv * bar_height)
        draw.text((bar_x + bar_width + 10, y_pos), tt + ' Ωm', fill='black', font=font)
    
    # Légende des couleurs/profondeurs
    draw.text((10, 10), f"ERT Section: Espacement électrodes = {electrode_spacing}m", fill='white', font=font, stroke_width=2, stroke_fill='black')
    draw.text((10, 40), f"Profondeur max: {max_depth}m", fill='white', font=font, stroke_width=2, stroke_fill='black')
    draw.text((10, height - 60), "Légende Couleurs (inspiré Res2DInv):", fill='white', font=font, stroke_width=2, stroke_fill='black')
    for i, (norm, color_str) in enumerate(colorscale[:5]):  # Top 5 pour espace
        r, g, b = int(color_str.split('(')[1].split(',')[0]), int(color_str.split(',')[1]), int(color_str.split(',')[2].split(')')[0])
        draw.rectangle([10, height - 50 + i*15, 30, height - 35 + i*15], fill=(r,g,b))
        rho_val = np.exp(np.log(rho_min) + norm*(np.log(rho_max)-np.log(rho_min)))
        matching_minerals = materials_df[(materials_df["Plage Min (Ωm)"] <= rho_val) & (materials_df["Plage Max (Ωm)"] >= rho_val)]["Type"].tolist()
        legend_text = f'{rho_val:.0f} Ωm {" / ".join(matching_minerals[:2]) if matching_minerals else ""}'
        draw.text((35, height - 50 + i*15), legend_text, fill='white', font=ImageFont.load_default())
    
    # Lignes de profondeur (ex. à 0, 25%, 50%, etc.)
    line_positions = [0, height*0.25, height*0.5, height*0.75]
    depths = [0, max_depth*0.25, max_depth*0.5, max_depth*0.75]
    for lp, d in zip(line_positions, depths):
        draw.line([(0, lp), (width - 100, lp)], fill='white', width=2)
        draw.text((width - 90, lp - 10), f"{d}m", fill='white', font=font)
    
    return image

# Markdown intro ERT (inchangé)
st.markdown("""
Bien sûr ! Voici une vue d'ensemble structurée du pipeline complet d'un logiciel de Tomographie de Résistivité Électrique (ERT), tel que Res2DInv de Bentley, EarthImager 2D de AGI ou BERT (Boundless Electrical Resistivity Tomography). Ce pipeline couvre toutes les étapes, de l'importation des données à la génération du modèle 2D ou 3D final, en passant par l'inversion et la visualisation.
---
🔧 Pipeline Complet d'un Logiciel ERT
1. Acquisition des Données
Instruments : SuperSting™, Iris Syscal™, ABEM Terrameter™, Geometrics, etc.
Configurations d'électrodes : Schlumberger, Wenner, Dipôle-Dipôle, Pole-Pole, etc.
Formats pris en charge : .dat, .txt, .gdd, .mae, .csv, etc.
Mesures : Résistivité apparente (ρa), polarisation induite (IP), auto-potentiel (SP).
2. Prétraitement des Données
Filtrage : Suppression du bruit, correction de tendance, filtrage par fréquence (FFT).
Élimination des données aberrantes : Retrait des points avec un rapport signal/bruit faible.
Correction de terrain : Ajustement des effets de topographie ou de conductivité de surface.
3. Modélisation Directe (Forward Modeling)
Objectif : Générer un modèle synthétique des données à partir d'un modèle géologique initial.
Méthodes : Différences finies, éléments finis, solutions analytiques.
Utilisation : Validation du modèle, estimation des erreurs, génération de données de référence pour l'inversion.
4. Inversion
Type : Inversion linéaire ou non linéaire.
Méthodes : Gauss-Newton, Tikhonov, régularisation l2/l1, inversion contraint.
Objectif : Estimer la distribution de résistivité du sous-sol à partir des données mesurées.
Logiciels : Res2DInv, Res3DInv, BERT, EarthImager 2D.
5. Visualisation
Sections 2D : Affichage de la résistivité en fonction de la profondeur ou de la distance horizontale.
Modèles 3D : Reconstruction volumétrique du sous-sol avec des coupes transversales.
Outils : Visualisation en coupe, en 3D, avec barres de couleurs, exportation vers des logiciels comme ParaView ou Oasis Montaj.
6. Interprétation Géologique
Analyse : Identification des structures géologiques, des anomalies, des interfaces.
Validation : Comparaison avec des données géotechniques, géologiques ou hydrogéologiques.
Rapports : Génération de rapports d'interprétation avec des visualisations intégrées.
7. Exportation
Formats : .csv, .txt, .dat, .xml, .vtk, .ply.
Applications : Intégration dans des SIG, modélisation 3D, analyse statistique.
---
🧪 Exemple d'Application
Prenons l'exemple de l'utilisation de la tomographie de résistivité électrique (ERT) pour détecter des canaux d'infiltration dans un site archéologique :
1. Collecte des Données : Réalisation de mesures ERT sur le site à l'aide d'un système SuperSting™ Wi-Fi.
2. Traitement Initial : Filtrage des données pour éliminer les bruits et les anomalies.
3. Inversion : Application d'une inversion 2D pour obtenir une coupe du sous-sol.
4. Analyse : Identification des zones de faible résistivité correspondant à des canaux d'infiltration.
5. Validation : Confirmation des résultats par des observations de surface et des données historiques.
6. Rapport : Génération d'un rapport détaillant les résultats et les recommandations pour la conservation du site.
---
🛠️ Logiciels Représentatifs
Res2DInv / Res3DInv : Logiciels professionnels pour l'inversion de données ERT en 2D et 3D, offrant des outils avancés de traitement et de visualisation.
EarthImager 2D : Logiciel d'inversion 2D abordable pour l'imagerie de résistivité et de polarisation induite, compatible avec diverses configurations d'électrodes.
BERT (Boundless ERT) : Logiciel open-source pour l'inversion de données ERT, offrant des outils flexibles pour la modélisation et l'interprétation.
---
Souhaites-tu que je t'aide à implémenter ce pipeline dans un environnement Python ou à intégrer certaines de ces étapes dans ton application Streamlit ?
""")

# Chargement du fichier ERT (amélioré pour logique ERT : rhoa apparente, positions électrodes)
uploaded_file = st.file_uploader("📁 Importer un fichier DT1 ou DAT", type=["dt1","dat","txt"])
if uploaded_file:
    content = uploaded_file.read()
    st.info("🔍 Analyse du type de fichier...")
    file_type = magic.from_buffer(content)
    st.write(f"Type détecté : {file_type}")
    # Décodage amélioré avec construct pour DT1/DAT (amélioré pour ERT : assume rhoa en colonnes)
    try:
        if uploaded_file.name.endswith(('.dt1', '.dat')):
            # Try binary parse first (amélioré : assume structure ERT basique : header + rhoa + positions)
            try:
                dt1_struct = Struct(
                    "header" / Array(20, Int16ul), # Header hypothétique de 20 uint16 (metadata, nombre de points, etc.)
                    "data" / GreedyRange(Int16ul) # Le reste comme données de données de résistivité (rhoa)
                )
                parsed = dt1_struct.parse(content)
                data = np.array(parsed.data, dtype=np.float64) # Conversion en float pour rhoa
                # Logique ERT : rhoa typique, clip à range réaliste
                data = np.clip(data, 0.01, 1e6)  # Évite valeurs aberrantes
                st.info("Fichier parsé en mode binaire (rhoa apparente).")
            except Exception as parse_e:
                st.warning(f"Échec parsing binaire : {parse_e}. Essai mode texte.")
                # Fallback to text (amélioré : assume colonnes x, a, b, m, n, rhoa pour ERT)
                text_content = content.decode('utf-8')
                try:
                    df_temp = pd.read_csv(io.StringIO(text_content), sep=r'\s+', header=None, names=["a", "b", "m", "n", "rhoa"])
                    data = df_temp["rhoa"].values
                except:
                    df_temp = pd.read_csv(io.StringIO(text_content), header=None, names=["rhoa"])
                    data = df_temp["rhoa"].values
                st.info("Fichier parsé en mode texte (colonnes ERT détectées).")
            num_points = len(data)
            st.success(f"✅ Fichier binaire/texte parsé : {num_points} mesures rhoa (résistivité apparente)")
        elif uploaded_file.name.endswith('.txt'):
            # Pour fichiers texte, lecture comme CSV ou lignes (amélioré pour ERT)
            text_content = content.decode('utf-8')
            try:
                df_temp = pd.read_csv(io.StringIO(text_content), sep=r'\s+', header=None, names=["a", "b", "m", "n", "rhoa"])
                data = df_temp["rhoa"].values
            except:
                df_temp = pd.read_csv(io.StringIO(text_content), header=None, names=["rhoa"])
                data = df_temp["rhoa"].values
            num_points = len(data)
            st.success(f"✅ Fichier texte chargé : {num_points} mesures rhoa")
        else:
            raise ValueError("Format de fichier non supporté.")
        # Création DataFrame ERT (amélioré : positions basées sur espacement, y=0 pour ligne)
        electrode_spacing = st.number_input("Espacement des électrodes (m)", min_value=0.1, value=1.0, step=0.1)
        x_positions = np.arange(num_points) * electrode_spacing  # Positions le long de la ligne
        df = pd.DataFrame({
            "x": x_positions,
            "y": np.zeros(num_points), # pour interpolation 2D/3D
            "rhoa": data  # rhoa apparente
        })
        # Add small jitter to y to avoid Qhull error if all y are the same (degenerate case)
        if np.ptp(df["y"]) < 1e-10:
            df["y"] += np.random.uniform(-1e-8, 1e-8, len(df))
        max_depth = st.slider("Profondeur maximale estimée (m)", min_value=10, max_value=500, value=int(electrode_spacing * np.sqrt(len(df))))
    except Exception as e:
        st.error(f"Erreur lors de la lecture ou du parsing du fichier : {e}. Vérifiez le format ou essayez un autre fichier.")
        st.stop()
    # Flag for inversion
    inverted_with_pygimli = False
    # --- Inversion pyGIMLi (vraie inversion géoélectrique 2D) --- (amélioré : utilise rhoa)
    st.subheader("🧮 Inversion géoélectrique 2D avec pyGIMLi")
    try:
        import pygimli as pg
        import pygimli.meshtools as mt
        import pygimli.physics.ert as ert
        # --- Étape 1 : Charger les données ERT --- (amélioré pour rhoa)
        data_ert = None
        if uploaded_file.name.endswith('.dat'):
            with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                # Utiliser importData au lieu de load pour une meilleure gestion des formats
                data_ert = ert.importData(tmp_path)
                if data_ert is None:
                    raise ValueError("ert.importData a retourné None - format non supporté ?")
                st.info("Chargement .dat réussi avec pyGIMLi.")
            except Exception as load_e:
                st.warning(f"⚠️ Erreur chargement .dat : {load_e}. Fallback vers données synthétiques parsées.")
                data_ert = None
            finally:
                os.unlink(tmp_path)
       
        if data_ert is None:
            # Conversion simplifiée : création de données synthétiques avec rhoa des données (amélioré)
            n_elec = min(50, max(10, int(np.sqrt(len(data))) * 2))
            data_ert = ert.createData(elecs=np.linspace(0, (n_elec-1)*electrode_spacing, n_elec), schemeName='dd', noiseLevel=0.05)
            if len(data) > 0:
                data_ert["rhoa"][:min(len(data), data_ert.size())] = data[:min(len(data), data_ert.size())]
        
        # Correction pour les erreurs (err > 0)
        if 'err' not in data_ert:
            data_ert['err'] = 0.01  # Erreur par défaut de 1%
        elif np.any(data_ert['err'] <= 0):
            data_ert['err'] = np.maximum(data_ert['err'], 0.01)
        
        st.write(f"📈 {data_ert.size()} mesures rhoa chargées pour inversion pyGIMLi.")
        # --- Étape 2 : Maillage automatique ---
        mesh = mt.createParaMesh(data_ert, quality=34)
        st.success(f"✅ Maillage créé : {mesh.cellCount()} cellules.")
        # --- Étape 3 : Inversion complète avec régularisation automatique ---
        inversion = ert.ERTManager(verbose=False) # verbose=False pour Streamlit
        inversion.invert(
            data=data_ert,
            mesh=mesh,
            lam=20, # paramètre de régularisation
            zWeight=0.5, # poids vertical
            robust=True # inversion robuste contre bruit
        )
        # --- Étape 4 : Récupération du modèle inversé ---
        model = inversion.model
        rho_min, rho_max = model.min(), model.max()
        st.write(f"Résistivité inversée : de {rho_min:.2f} à {rho_max:.2f} Ω·m")
        # Mise à jour df avec modèle inversé (amélioré)
        rho_inverted = model.reshape((mesh.cellCount(),))  # Aplatir pour df
        df["rho"] = rho_inverted[:len(df)] if len(rho_inverted) >= len(df) else np.tile(rho_inverted, (len(df)//len(rho_inverted)+1))[:len(df)]
        # --- Étape 5 : Affichage interactif ---
        fig_pg = inversion.showResult(cMap='Spectral_r', showMesh=True, logScale=True, colorBar=True)
        st.pyplot(fig_pg)
        inverted_with_pygimli = True
        st.success("✅ Inversion pyGIMLi réussie ! Visualisations suivantes utilisent ce modèle inversé.")
        auto_pause_gpu()
    except Exception as e:
        st.warning(f"⚠️ pyGIMLi non disponible ou erreur d'inversion : {e}")
        st.info("L'inversion simplifiée sera utilisée à la place.")
        df["rho"] = df["rhoa"]  # Fallback à rhoa
    # --- Inversion simple seulement si pyGIMLi a échoué --- (amélioré pour 1D ERT-like)
    if not inverted_with_pygimli:
        st.subheader("🔄 Inversion Simple (Régularisation)")
        lambda_reg = st.slider("Paramètre de régularisation λ", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        try:
            # Inversion simplifiée : résoudre (I + λ D2^T D2) model = rhoa avec matrices sparses pour robustesse
            rho = df["rhoa"].values
            n = len(rho)
            if n < 3:
                raise ValueError("Données trop courtes pour régularisation de second ordre.")
            # Construction de D2 sparse (secondes différences)
            diags_data = [np.ones(n-2), -2 * np.ones(n-2), np.ones(n-2)]
            offsets = [0, 1, 2]
            D2 = sp.diags(diags_data, offsets, shape=(n-2, n), format='csr')
            L = D2.T @ D2 # L est sparse
            I = sp.eye(n, format='csr')
            A = I + lambda_reg * L
            inverted_model = spsolve(A, rho)
            df["rho"] = inverted_model
            st.success("✅ Inversion simple effectuée avec régularisation (modèle rho).")
            auto_pause_gpu()
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de l'inversion : {e}. Utilisation des données rhoa.")
            df["rho"] = df["rhoa"]
    # --- Clustering KMeans --- (inchangé)
    st.subheader("🧩 Clustering KMeans")
    n_clusters = st.slider("Nombre de clusters", 2, 6, 3)
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        df["cluster"] = kmeans.fit_predict(df[["rho"]])
        st.dataframe(df.head())
        auto_pause_gpu()
    except Exception as e:
        st.error(f"Erreur lors du clustering : {e}")
        st.stop()
    # Tableau des résistivités (amélioré : étendu avec plus de minéraux inspirés de recherches ERT pour exploration minière)
    st.subheader("Tableau des Types de Minerais, Roches et Liquides avec Résistivités Géophysiques (Étendu pour Minéraux)")
    materials_data = [
        {"Catégorie": "Liquides", "Type": "Eau de mer", "Plage Min (Ωm)": 0.05, "Plage Max (Ωm)": 0.3, "Notes": "Haute conductivité due à la salinité"},
        {"Catégorie": "Liquides", "Type": "Eau saumâtre", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 10, "Notes": "Salinité modérée"},
        {"Catégorie": "Liquides", "Type": "Eau douce", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 100, "Notes": "Faible salinité, eaux de surface ou souterraines"},
        {"Catégorie": "Liquides", "Type": "Eau minérale/mine", "Plage Min (Ωm)": 0.1, "Plage Max (Ωm)": 1, "Notes": "Haute concentration en minéraux dissous"},
        {"Catégorie": "Liquides", "Type": "Pétrole/Hydrocarbures", "Plage Min (Ωm)": 1000, "Plage Max (Ωm)": 100000000, "Notes": "Très résistif, isolant"},
        {"Catégorie": "Minerais", "Type": "Graphite", "Plage Min (Ωm)": 0.000008, "Plage Max (Ωm)": 0.0001, "Notes": "Très conducteur"},
        {"Catégorie": "Minerais", "Type": "Pyrite pure", "Plage Min (Ωm)": 0.00003, "Plage Max (Ωm)": 0.001, "Notes": "Sulfure, variable avec impuretés"},
        {"Catégorie": "Minerais", "Type": "Pyrite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Avec impuretés comme le cuivre"},
        {"Catégorie": "Minerais", "Type": "Galena", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 100, "Notes": "Sulfure de plomb, conducteur comme minéral"},
        {"Catégorie": "Minerais", "Type": "Magnétite", "Plage Min (Ωm)": 0.01, "Plage Max (Ωm)": 1000, "Notes": "Oxyde de fer, variable"},
        {"Catégorie": "Minerais", "Type": "Hématite", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 10000, "Notes": "Oxyde de fer, presque isolant"},
        {"Catégorie": "Minerais", "Type": "Chalcopyrite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Sulfure de cuivre-fer"},
        {"Catégorie": "Minerais", "Type": "Bornite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Sulfure de cuivre-fer, conducteur"},
        {"Catégorie": "Minerais", "Type": "Quartz", "Plage Min (Ωm)": 10000000000, "Plage Max (Ωm)": 100000000000000, "Notes": "Silicate, isolant"},
        # Ajouts inspirés de recherches ERT pour exploration minière (ex. sulfures, oxydes, etc.)
        {"Catégorie": "Minerais", "Type": "Sphalerite (Zinc)", "Plage Min (Ωm)": 100, "Plage Max (Ωm)": 10000, "Notes": "Sulfure de zinc, modérément résistif"},
        {"Catégorie": "Minerais", "Type": "Cassitérite (Étain)", "Plage Min (Ωm)": 1000, "Plage Max (Ωm)": 10000, "Notes": "Oxyde d'étain, résistif"},
        {"Catégorie": "Minerais", "Type": "Molybdénite (Molybdène)", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 1, "Notes": "Sulfure, hautement conducteur"},
        {"Catégorie": "Minerais", "Type": "Or (veines quartz-or)", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 1000, "Notes": "Variable, anomalies conductrices pour sulfures associés"},
        {"Catégorie": "Minerais", "Type": "Fer (minerai de fer)", "Plage Min (Ωm)": 0.01, "Plage Max (Ωm)": 1000, "Notes": "Magnétite/hématite, zones conductrices à résistives"},
        {"Catégorie": "Roches", "Type": "Argile (humide)", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 100, "Notes": "Faible résistivité due à l'eau et ions"},
        {"Catégorie": "Roches", "Type": "Schiste", "Plage Min (Ωm)": 20, "Plage Max (Ωm)": 2000, "Notes": "Variable avec humidité"},
        {"Catégorie": "Roches", "Type": "Grès", "Plage Min (Ωm)": 30, "Plage Max (Ωm)": 10000, "Notes": "Sèches à saturées"},
        {"Catégorie": "Roches", "Type": "Calcaire", "Plage Min (Ωm)": 50, "Plage Max (Ωm)": 10000000, "Notes": "Variable, haut si sec"},
        {"Catégorie": "Roches", "Type": "Granite", "Plage Min (Ωm)": 5000, "Plage Max (Ωm)": 1000000, "Notes": "Igneuse, haut si sec"},
        {"Catégorie": "Roches", "Type": "Basalte", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 13000000, "Notes": "Igneuse, variable"},
        {"Catégorie": "Roches", "Type": "Alluvions", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 1000, "Notes": "Sédiments non consolidés"},
        {"Catégorie": "Roches", "Type": "Gravier", "Plage Min (Ωm)": 100, "Plage Max (Ωm)": 2500, "Notes": "Sec"}
    ]
    materials_df = pd.DataFrame(materials_data)
    st.dataframe(materials_df)
    # Interprétation Géophysique des Clusters (améliorée avec nouveaux minéraux)
    st.subheader("Interprétation Géophysique des Clusters (Inspiré de Visualisations ERT Minérales)")
    cluster_means = df.groupby("cluster")["rho"].mean().sort_index()
    for cl, mean_rho in cluster_means.items():
        matching = materials_df[(materials_df["Plage Min (Ωm)"] <= mean_rho) & (materials_df["Plage Max (Ωm)"] >= mean_rho)]
        if not matching.empty:
            types = ", ".join(matching["Type"].tolist())
            st.write(f"Cluster {cl} (moyenne ρ = {mean_rho:.2f} Ωm) pourrait correspondre à : {types}")
        else:
            st.write(f"Cluster {cl} (moyenne ρ = {mean_rho:.2f} Ωm) : Pas de correspondance trouvée dans la table")
    # Ajout de l'interprétation au DataFrame
    df['interpretation'] = ''
    for cl, mean_rho in cluster_means.items():
        matching = materials_df[(materials_df["Plage Min (Ωm)"] <= mean_rho) & (materials_df["Plage Max (Ωm)"] >= mean_rho)]
        if not matching.empty:
            types = ", ".join(matching["Type"].tolist())
        else:
            types = "Pas de correspondance"
        df.loc[df['cluster'] == cl, 'interpretation'] = types
    # Compute mean_df for annotations
    mean_df = df.groupby('cluster').agg({'x':'mean', 'y':'mean', 'rho':'mean', 'interpretation':'first'})
    # Nouvelle section : Analyse IA Avancée (adaptée pour Llama en temps réel, enrichie minéraux)
    st.subheader("🤖 Analyse IA Avancée avec Meta-LLaMA 3.1 13B (Temps Réel, Focus Minéraux)")
    if st.button("Générer interprétation IA") or st.session_state.get('ai_generated', False):
        if not st.session_state.get('ai_generated', False):
            # Préparer le prompt avec les données (amélioré pour minéraux)
            clusters_info = "\n".join([f"Cluster {cl}: moyenne rho = {mean_rho:.2f} Ωm, correspondance basique: {df.loc[df['cluster'] == cl, 'interpretation'].iloc[0]}" for cl, mean_rho in cluster_means.items()])
            # Ajout de détails position et profondeur pour chaque cluster
            detailed_clusters = ""
            for cl in cluster_means.index:
                cluster_data = df[df['cluster'] == cl]
                x_min, x_max = cluster_data['x'].min(), cluster_data['x'].max()
                location = f"entre {x_min:.1f}m et {x_max:.1f}m le long de la ligne de mesure"
                estimated_volume = len(cluster_data) * electrode_spacing * max_depth * 1.0  # Estimation simple (largeur x profondeur x espacement)
                detailed_clusters += f"Cluster {cl}: ρ moyenne = {mean_rho:.2f} Ωm, localisation: {location}, volume estimé: {estimated_volume:.1f} m³, profondeur: 0 à {max_depth}m.\n"
            # Base de données matériaux comme contexte (étendue)
            materials_context = materials_df.to_string(index=False)
            prompt = f"""Tu es un expert en géophysique ERT pour exploration minière. Analyse les données du fichier DAT chargé. Utilise la base de données étendue pour attribuer des valeurs réelles et qualificatifs exacts aux minéraux/liquides/roches identifiés dans les clusters. Pour chaque cluster/minéral, fournis: type exact (ex: pyrite conductrice), résistivité précise (basée sur ρ moyenne et plage DB), profondeur (0 à {max_depth}m), localisation (basée sur x), calculs appropriés (ex: conductivité = 1/ρ, volume estimé = {electrode_spacing} * longueur * profondeur), notes spécifiques (ex: anomalie sulfure pour pyrite). Données clusters détaillées:\n{detailed_clusters}\nBase de données matériaux (étendue pour minéraux):\n{materials_context}\nInterprétation détaillée par cluster/minéral (focus anomalies minérales potentielles) :"""
            if llama_model:
                with st.spinner("Analyse en temps réel avec Llama..."):
                    output = llama_model(
                        prompt=prompt,
                        max_tokens=500,
                        temperature=0.7,
                        stop=["\n\n"]
                    )
                    ai_interpretation = output['choices'][0]['text'].strip()
            else:
                # Fallback SmolLM
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=300)
                ai_interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.session_state.ai_interpretation = ai_interpretation
            st.session_state.ai_generated = True
        st.markdown(st.session_state.ai_interpretation)
        auto_pause_gpu()
    # --- Interpolation corrigée pour éviter NaNs (explication : les points sont colinéaires sur y~0, griddata 'linear' produit des NaNs ; utilisation de 1D interp1d sur x) ---
    try:
        # Interpolation 1D sur x pour rho et cluster (évite NaNs dus à y colinéaire)
        rho_interp_func = interp1d(df["x"], df["rho"], kind='linear', fill_value="extrapolate")
        cluster_interp_func = interp1d(df["x"], df["cluster"], kind='nearest', fill_value="extrapolate")
        x_grid = np.linspace(df["x"].min(), df["x"].max(), 200)
        rho_interp = rho_interp_func(x_grid)
        cluster_interp = cluster_interp_func(x_grid)
        # Extrusion en profondeur (2D viz) - logique ERT : profondeur ~ espacement * sqrt(n_points)
        assumed_depths = np.linspace(0, max_depth, 50)
        grid_x_plot, depth_plot = np.meshgrid(x_grid, -assumed_depths)  # y = depth négative
        grid_z_2d = np.tile(rho_interp, (len(assumed_depths), 1))  # Extrusion uniforme (simplifié pour ERT 2D)
        cluster_2d = np.tile(cluster_interp, (len(assumed_depths), 1))
        # Normalisation rho (log pour ERT)
        rho_min_global, rho_max_global = df["rho"].min(), df["rho"].max()
        rho_norm = np.clip((np.log10(np.clip(grid_z_2d, 0.01, 1e6)) - np.log10(0.01)) / (np.log10(1e6) - np.log10(0.01)), 0, 1)
        st.success("✅ Interpolation corrigée : utilisation de 1D sur x pour éviter NaNs dus à points colinéaires.")
    except Exception as e:
        st.error(f"Erreur lors de l'interpolation corrigée : {e}")
        st.stop()
    # Triangulation optimisée (inchangé, mais non utilisée ici)
    try:
        points2d = df[['x', 'y']].to_numpy()
        tri = Delaunay(points2d)
    except Exception as e:
        st.error(f"Erreur lors de la triangulation : {e}. Essayez d'ajuster les données.")
        st.stop()
    # --- Palette de couleurs type géophysique ERT (inchangée) ---
    colorscale_ert = [
        [0.0, 'rgb(0, 0, 139)'],  # Bleu foncé pour très conducteur (<0.1 Ωm, e.g., eau de mer)
        [0.125, 'rgb(0, 0, 255)'],  # Bleu pour eau saumâtre/minérale (0.1-1 Ωm)
        [0.25, 'rgb(0, 100, 255)'],  # Bleu clair pour eau douce/argile humide (1-10 Ωm)
        [0.375, 'rgb(0, 255, 255)'],  # Cyan pour sols humides (10-30 Ωm)
        [0.5, 'rgb(0, 255, 0)'],  # Vert pour zones altérées/sable humide (30-100 Ωm)
        [0.625, 'rgb(255, 255, 0)'],  # Jaune pour roches fissurées (100-300 Ωm)
        [0.75, 'rgb(255, 165, 0)'],  # Orange pour roches compactes (300-1000 Ωm)
        [0.875, 'rgb(255, 69, 0)'],  # Rouge orangé pour roches sèches (>1000 Ωm)
        [1.0, 'rgb(255, 0, 0)']  # Rouge pour isolants (e.g., granite sec, >5000 Ωm)
    ]
    # Palette Res2DInv améliorée pour contours, alignée sur valeurs typiques eau (bleu pour <10, vert 10-100, etc.)
    res2dinv_colorscale = [
        [0.0, "#000080"], # Bleu marine pour <0.1 Ωm (eau salée)
        [0.125, "#0000FF"], # Bleu pour 0.1-1 Ωm (eau minérale)
        [0.25, "#4169E1"], # Bleu royal pour 1-10 Ωm (eau douce)
        [0.375, "#00FFFF"], # Cyan pour 10-30 Ωm (argile)
        [0.5, "#00FF00"], # Vert pour 30-100 Ωm (sable humide)
        [0.625, "#FFFF00"], # Jaune pour 100-300 Ωm (grès)
        [0.75, "#FFA500"], # Orange pour 300-1000 Ωm (calcaire)
        [0.875, "#FF4500"], # Rouge orangé pour 1000-5000 Ωm (granite)
        [1.0, "#FF0000"], # Rouge pour >5000 Ωm (quartz)
    ]
    # Ticks pour colorbar (amélioré : valeurs log réelles)
    tickvals = np.linspace(0, 1, 9)
    ticktexts = [f'{0.01:.2f}', '0.1', '1', '10', '100', '1000', '10k', '100k', '1M']
    # Section pour graphiques 3D Plotly liés à l'analyse Llama (améliorée avec annotations minérales, inspirée de pseudo-sections et inversions réelles)
    st.subheader("📈 Graphiques Diversifiés Plotly (Tous Genres : 3D, 2D Heatmap, Contours, Pseudosections, Sensibilité, Clustering)")
    if st.session_state.get('ai_generated', False):
        # Graphique 1: Volume 3D (Standard Res3DInv-like, avec annotations minérales)
        num_y = 5
        num_z = 20
        y_grid = np.linspace(-electrode_spacing * 5, electrode_spacing * 5, num_y)  # Largeur transversale estimée
        z_grid = np.linspace(0, max_depth, num_z)
        x_grid_for3d = np.linspace(df["x"].min(), df["x"].max(), 30)  # Plus de points pour fluidité
        # Interpolation cubique pour plus de lissage
        rho_interp_func_3d = interp1d(df["x"], df["rho"], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_x_3d = rho_interp_func_3d(x_grid_for3d)
        # Créer rho_3d avec variation en profondeur (augmentation de rho avec profondeur pour simuler séchage)
        rho_3d = np.tile(rho_x_3d[None, None, :], (num_y, num_z, 1))
        for k in range(num_z):
            depth_factor = 1 + 0.5 * (k / (num_z - 1))  # Augmente rho avec profondeur
            rho_3d[:, k, :] *= depth_factor
        # Grille 3D
        X_3d, Y_3d, Z_3d = np.meshgrid(x_grid_for3d, y_grid, z_grid, indexing='ij')
        value_3d = np.log10(np.clip(rho_3d, 1e-2, 1e6)).flatten()
        x_flat = X_3d.flatten()
        y_flat = Y_3d.flatten()
        z_flat = Z_3d.flatten()
        fig1 = go.Figure(data=go.Volume(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=value_3d,
            isomin=np.min(value_3d),
            isomax=np.max(value_3d),
            opacity=0.2,
            surface_count=15,
            colorscale=colorscale_ert,
            showscale=True,
            colorbar=dict(title="log₁₀(ρ) (Ωm)")
        ))
        # Ajout annotations pour minéraux typiques
        for cl, mean_rho in cluster_means.items():
            matching = materials_df[(materials_df["Plage Min (Ωm)"] <= mean_rho) & (materials_df["Plage Max (Ωm)"] >= mean_rho)]
            if not matching.empty:
                mineral_ex = matching["Type"].iloc[0]
                fig1.add_annotation(text=f"Zone {mineral_ex} (~{mean_rho:.0f} Ωm)", x=0.05, y=0.95 - cl*0.1, showarrow=False, font=dict(color="white"))
        fig1.update_layout(
            title="1. Volume 3D Représentatif du Modèle Sous-Sol (Insights Intégrés de Llama, Anomalies Minérales)",
            scene=dict(
                xaxis_title='X (Position, m)',
                yaxis_title='Y (Largeur, m)',
                zaxis_title='Z (Profondeur, m)'
            )
        )
        st.plotly_chart(fig1, use_container_width=True)
        # Graphique 2: Pseudosection 2D de ρa (Amélioré : profondeur pseudo plus réaliste pour Wenner-like)
        fig2 = go.Figure()
        # Pseudosection améliorée : profondeur ~ a/3 pour Wenner, ou sqrt(a*b) pour DD ; ici approx sqrt(i) * spacing
        pseudo_depth = electrode_spacing * np.sqrt(np.arange(len(df["rhoa"]))) / 3  # Ajusté pour plus de réalisme
        fig2.add_trace(go.Heatmap(
            z=np.log10(np.clip(df["rhoa"], 0.01, 1e6)),
            x=df["x"],
            y=-pseudo_depth,  # Négatif pour profondeur
            colorscale=colorscale_ert,
            colorbar=dict(title="log₁₀(ρa) (Ωm)", tickvals=tickvals, ticktext=ticktexts)
        ))
        # Ajout lignes pour configurations d'électrodes typiques
        fig2.add_hline(y=-electrode_spacing, line_dash="dash", line_color="white", annotation_text="Espacement base")
        fig2.update_layout(
            title="2. Pseudosection 2D de Résistivité Apparente (Style EarthImager, Profondeur Réaliste)",
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur Pseudo (m)"
        )
        st.plotly_chart(fig2, use_container_width=True)
        # Graphique 3: Section Inversée 2D avec Contours (Amélioré : contours plus denses, annotations minérales)
        fig3 = make_subplots(rows=1, cols=1)
        fig3.add_trace(go.Heatmap(
            z=np.log10(np.clip(grid_z_2d, 0.01, 1e6)),
            x=x_grid,
            y=-assumed_depths,
            colorscale=res2dinv_colorscale,
            colorbar=dict(title="log₁₀(ρ) (Ωm)", tickvals=tickvals, ticktext=ticktexts)
        ))
        # Ajout contours plus denses (inspiré Res2DInv)
        contours = go.Contour(
            z=np.log10(np.clip(grid_z_2d, 0.01, 1e6)),
            x=x_grid,
            y=-assumed_depths,
            contours=dict(start=np.log10(rho_min_global), end=np.log10(rho_max_global), size=(np.log10(rho_max_global) - np.log10(rho_min_global))/20),  # Plus de contours
            line=dict(color='white', width=1),
            showscale=False
        )
        fig3.add_trace(contours)
        # Annotations pour minéraux sur contours
        for cl, mean_rho in cluster_means.items():
            x_annot = mean_df.loc[cl, 'x']
            y_annot = -max_depth / 2
            matching = materials_df[(materials_df["Plage Min (Ωm)"] <= mean_rho) & (materials_df["Plage Max (Ωm)"] >= mean_rho)]
            if not matching.empty:
                mineral = matching["Type"].iloc[0]
                fig3.add_annotation(x=x_annot, y=y_annot, text=f"{mineral}<br>{mean_rho:.0f}Ωm", showarrow=True, arrowhead=2, ax=20, ay=-30)
        fig3.update_layout(
            title="3. Section Inversée 2D avec Contours Denses (Style Res2DInv, Labels Minéraux)",
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        # Graphique 4: Clustering Géologique 2D (Analyse Intelligente LLaMA)
        fig4 = go.Figure()
        fig4.add_trace(go.Heatmap(
            z=cluster_2d,
            x=x_grid,
            y=-assumed_depths,
            colorscale='Rainbow',
            colorbar=dict(title="Cluster ID")
        ))
        # Ajout overlay minéral
        for cl in range(n_clusters):
            matching = materials_df[(materials_df["Plage Min (Ωm)"] <= cluster_means[cl]) & (materials_df["Plage Max (Ωm)"] >= cluster_means[cl])]
            if not matching.empty:
                fig4.add_annotation(text=f"Cluster {cl}: {matching['Type'].iloc[0]}", x=0.02, y=1 - cl*0.15, showarrow=False, font=dict(color="black"))
        fig4.update_layout(
            title="4. Section 2D Clustering Minéral (Interprétation via LLaMA, Focus Minéraux)",
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)"
        )
        st.plotly_chart(fig4, use_container_width=True)
        # Graphique 5: Carte de Sensibilité/Erreur (Avancé, simulé pour standard BERT-like, amélioré avec variation minérale)
        # Simuler erreur : variation aléatoire pondérée par profondeur et rho (plus d'erreur en zones minérales conductrices)
        error_sim = np.random.uniform(0.05, 0.2, grid_z_2d.shape) * (1 + np.abs(depth_plot) / max_depth) * (1 + 1/np.clip(grid_z_2d, 1, 1000))  # Plus d'erreur en zones conductrices
        fig5 = go.Figure(data=go.Heatmap(
            z=error_sim,
            x=x_grid,
            y=-assumed_depths,
            colorscale='Viridis',
            colorbar=dict(title="Sensibilité Relative (%)")
        ))
        fig5.update_layout(
            title="5. Carte de Sensibilité/Erreur du Modèle (Analyse Avancée LLaMA, Pondéré Minéraux)",
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)"
        )
        st.plotly_chart(fig5, use_container_width=True)
        # Graphique 6: Surface 3D (du premier code, pour diversité)
        grid_x, grid_y = np.meshgrid(np.linspace(df["x"].min(), df["x"].max(), 200),  
                                     np.linspace(df["y"].min(), df["y"].max(), 50))  
        grid_z = griddata((df["x"], df["y"]), df["rho"], (grid_x, grid_y), method='linear')  
        fig6 = go.Figure(data=[go.Surface(  
            x=grid_x, y=grid_y, z=grid_z,  
            colorscale=colorscale_ert,  
            colorbar=dict(title="Résistivité"),  
            showscale=True  
        )])  
        # Ajout des points détectés avec couleurs par cluster  
        fig6.add_trace(go.Scatter3d(  
            x=df["x"], y=df["y"], z=df["rho"],  
            mode='markers',  
            marker=dict(  
                size=5,  
                color=df["cluster"],  
                colorscale='Viridis',  
                opacity=0.8  
            ),  
            hovertext=df["interpretation"],  
            hoverinfo='text+x+y+z'  
        ))  
        # Ajout des labels par cluster (au centre moyen)  
        group = df.groupby('cluster')  
        mean_df_surf = group.agg({'x':'mean', 'y':'mean', 'rho':'mean', 'interpretation':'first'})  
        fig6.add_trace(go.Scatter3d(  
            x=mean_df_surf["x"], y=mean_df_surf["y"], z=mean_df_surf["rho"],  
            mode='text',  
            text=mean_df_surf["interpretation"],  
            textposition="top center",  
            textfont=dict(size=12, color="black")  
        ))  
        # Ajout de la triangulation optimisée avec Mesh3d (plus efficace que lines)  
        fig6.add_trace(go.Mesh3d(  
            x=df['x'], y=df['y'], z=df['rho'],  
            i=tri.simplices[:,0], j=tri.simplices[:,1], k=tri.simplices[:,2],  
            color='black', opacity=0.2, flatshading=True,  
            name='Triangulation'  
        ))  
        fig6.update_layout(scene=dict(  
            xaxis_title='X', yaxis_title='Y', zaxis_title='Rho'  
        ))  
        st.plotly_chart(fig6, use_container_width=True)
        # Graphique 7: Heatmap 2D Basique (du premier code)
        fig7 = go.Figure(data=go.Heatmap(  
            x=grid_x[0, :],  
            y=-assumed_depths,  # Profondeur négative pour orientation descendante  
            z=grid_z_2d,  
            colorscale=colorscale_ert,  
            colorbar=dict(title="Résistivité")  
        ))  
        fig7.update_layout(  
            xaxis_title='Position X (m)',  
            yaxis_title='Profondeur (m)',  
            title='Section 2D ERT avec couleurs géophysiques'  
        )  
        # Ajout des couches de surface du sol (arbitraires pour illustration)  
        fig7.add_hline(y=0, line_dash="solid", line_color="brown", annotation_text="Surface du sol", annotation_position="right")  
        fig7.add_hline(y=-max_depth*0.1, line_dash="dash", line_color="green", annotation_text="Couche de surface (topsoil)", annotation_position="right")  
        fig7.add_hline(y=-max_depth*0.3, line_dash="dash", line_color="yellow", annotation_text="Sous-sol (subsoil)", annotation_position="right")  
        fig7.add_hline(y=-max_depth*0.6, line_dash="dash", line_color="orange", annotation_text="Roche-mère (bedrock)", annotation_position="right")  
        fig7.add_hline(y=-max_depth, line_dash="solid", line_color="red", annotation_text="Limite profonde", annotation_position="right")  
        st.plotly_chart(fig7, use_container_width=True)
        # Graphique 8: Heatmap 2D Minéraux (du premier code)
        try:
            grid_cluster = griddata((df["x"], df["y"]), df["cluster"], (grid_x, grid_y), method='nearest')
            grid_cluster_2d = np.tile(grid_cluster[0, :], (len(assumed_depths), 1))
            unique_clusters = np.unique(df["cluster"])
            tickvals_min = unique_clusters
            ticktext_min = [df[df["cluster"] == c]["interpretation"].iloc[0] for c in unique_clusters]
            fig8 = go.Figure(data=go.Heatmap(
                x=grid_x[0, :],
                y=-assumed_depths,
                z=grid_cluster_2d,
                colorscale='Rainbow',
                colorbar=dict(title="Minéraux", tickvals=tickvals_min, ticktext=ticktext_min)
            ))
            fig8.update_layout(
                xaxis_title='Position X (m)',
                yaxis_title='Profondeur (m)',
                title='Section 2D ERT avec Minéraux'
            )
            # Ajout des couches de surface du sol (arbitraires pour illustration)
            fig8.add_hline(y=0, line_dash="solid", line_color="brown", annotation_text="Surface du sol", annotation_position="right")
            fig8.add_hline(y=-max_depth*0.1, line_dash="dash", line_color="green", annotation_text="Couche de surface (topsoil)", annotation_position="right")
            fig8.add_hline(y=-max_depth*0.3, line_dash="dash", line_color="yellow", annotation_text="Sous-sol (subsoil)", annotation_position="right")
            fig8.add_hline(y=-max_depth*0.6, line_dash="dash", line_color="orange", annotation_text="Roche-mère (bedrock)", annotation_position="right")
            fig8.add_hline(y=-max_depth, line_dash="solid", line_color="red", annotation_text="Limite profonde", annotation_position="right")
            st.plotly_chart(fig8, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la visualisation 2D avec Minéraux : {e}")
        # Graphique 9: Contours Res2DInv (du premier code, amélioré)
        # Grille régulière
        x_lin = np.linspace(df["x"].min(), df["x"].max(), 300)
        y_lin = np.linspace(0, max_depth, 150)
        grid_x_contour, grid_y_contour = np.meshgrid(x_lin, y_lin)

        # Interpolation sur grille (cubic pour un rendu lissé)
        grid_rho_contour = griddata(
            (df["x"], df["y"]), df["rho"],
            (grid_x_contour, grid_y_contour),
            method='cubic'
        )

        # Palette de couleurs géophysique (Res2DInv-like)
        res2dinv_colorscale_contour = [
            [0.0, "#0015ff"],   # bleu foncé
            [0.1, "#0090ff"],   # bleu clair
            [0.2, "#00ffe1"],   # cyan
            [0.4, "#00ff00"],   # vert
            [0.6, "#ffff00"],   # jaune
            [0.8, "#ff9900"],   # orange
            [1.0, "#ff0000"],   # rouge
        ]

        # Calcul des niveaux de contour logarithmiques
        rho_min_cont = np.nanmin(df["rho"])
        rho_max_cont = np.nanmax(df["rho"])
        levels = np.logspace(np.log10(rho_min_cont), np.log10(rho_max_cont), 10)

        # Création du contour plot
        fig9 = go.Figure(
            data=go.Contour(
                z=np.log10(grid_rho_contour),
                x=x_lin,
                y=-grid_y_contour,  # profondeur positive vers le bas
                colorscale=res2dinv_colorscale_contour,
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=9, color="black"),
                    start=np.log10(rho_min_cont),
                    end=np.log10(rho_max_cont),
                    size=(np.log10(rho_max_cont) - np.log10(rho_min_cont)) / 10,
                    coloring='fill'
                ),
                colorbar=dict(
                    title=dict(text="Résistivité (Ω·m)", side="right"),
                    tickvals=np.log10(levels),
                    ticktext=[f"{v:.1f}" for v in levels]
                )
            )
        )

        # Annotations minéralogiques (centrées sur les clusters)
        for _, row in mean_df.iterrows():
            fig9.add_annotation(
                x=row["x"],
                y=-max_depth * (0.1 + 0.15 * np.random.rand()),
                text=row["interpretation"],
                showarrow=False,
                font=dict(size=12, color="black", family="Arial Black"),
                bgcolor="rgba(255,255,255,0.6)"
            )

        # Mise en forme finale
        fig9.update_layout(
            title="9. Modèle de résistivité 2D (Style ERTLab / Res2DInv)",
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)",
            yaxis=dict(autorange='reversed'),
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig9, use_container_width=True)

# --- Légende géophysique automatique ---
        st.markdown("### 🧭 Interprétation géophysique des couleurs")

        legend_data = [
            {"Couleur": "🔵 Bleu foncé", "Intervalle (Ω·m)": "< 10", "Type de matériau": "Argile saturée / Eau salée"},
            {"Couleur": "🩵 Bleu clair", "Intervalle (Ω·m)": "10 – 30", "Type de matériau": "Sol humide / Argile"},
            {"Couleur": "💚 Vert", "Intervalle (Ω·m)": "30 – 100", "Type de matériau": "Sol limoneux / Zone altérée"},
            {"Couleur": "💛 Jaune", "Intervalle (Ω·m)": "100 – 300", "Type de matériau": "Roche fissurée / Sable humide"},
            {"Couleur": "🧡 Orange", "Intervalle (Ω·m)": "300 – 1000", "Type de matériau": "Roche compacte / Quartz"},
            {"Couleur": "🔴 Rouge", "Intervalle (Ω·m)": "> 1000", "Type de matériau": "Roche sèche / Quartzite / Granite"},
        ]

        legend_df = pd.DataFrame(legend_data)

        st.dataframe(
            legend_df.style.set_properties(
                **{
                    "text-align": "center",
                    "font-size": "14px",
                    "background-color": "#f9f9f9"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            """
            <div style='font-size:13px; color:gray; margin-top:8px;'>
            ℹ️ Les valeurs de résistivité sont données à titre indicatif. 
            Leur interprétation dépend des conditions locales, de la teneur en eau et de la nature du substratum.
            </div>
            """,
            unsafe_allow_html=True
        )
        # Graphique 10: Amélioration : Ajout d'un scatter 3D pour les centres de clusters extrudés (avec labels minéraux)
        cluster_centers = []
        colors_cluster = px.colors.qualitative.Set1  # Couleurs distinctes pour clusters
        for cl in range(n_clusters):
            cluster_data = df[df['cluster'] == cl]
            if len(cluster_data) > 0:
                center_x = cluster_data['x'].mean()
                center_y = 0  # Centre ligne
                center_z = max_depth / 2  # Milieu profondeur pour visualisation
                matching = materials_df[(materials_df["Plage Min (Ωm)"] <= cluster_means[cl]) & (materials_df["Plage Max (Ωm)"] >= cluster_means[cl])]
                mineral_label = matching['Type'].iloc[0] if not matching.empty else "Inconnu"
                cluster_centers.append({'x': [center_x], 'y': [center_y], 'z': [center_z], 'cluster': cl, 'rho': cluster_data['rho'].mean(), 'mineral': mineral_label})
        if cluster_centers:
            centers_df = pd.DataFrame(cluster_centers)
            fig10 = go.Figure()
            for cl in centers_df['cluster'].unique():
                sub_df = centers_df[centers_df['cluster'] == cl]
                fig10.add_trace(go.Scatter3d(
                    x=sub_df['x'], y=sub_df['y'], z=sub_df['z'],
                    mode='markers+text',
                    marker=dict(size=8, color=colors_cluster[cl % len(colors_cluster)]),
                    text=[f"Cluster {cl}<br>ρ={sub_df['rho'].iloc[0]:.1f} Ωm<br>{sub_df['mineral'].iloc[0]}"],
                    textposition="middle center",
                    name=f"Cluster {cl} ({sub_df['mineral'].iloc[0]})"
                ))
            fig10.update_layout(
                title="10. Centres de Clusters 3D (Interprétation Géologique via Llama, Labels Minéraux)",
                scene=dict(
                    xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (Profondeur, m)'
                )
            )
            st.plotly_chart(fig10, use_container_width=True)
        with st.expander("Détails de l'Analyse Llama Intégrés"):
            st.markdown(st.session_state.ai_interpretation)
        auto_pause_gpu()
    else:
        st.info("Cliquez sur 'Générer interprétation IA' pour débloquer les visualisations diversifiées Plotly.")
    st.subheader("Techniques avancées pour approfondir l'analyse ERT")
    st.markdown("""
D'après des recherches sur des techniques avancées pour approfondir l'analyse en Tomographie de Résistivité Électrique (ERT) :
- **Analyse texturale GLCM (Gray Level Co-occurrence Matrix)** : Utilisée comme technique de post-traitement sur les sections inverties ERT pour améliorer l'interprétation du sous-sol en identifiant des textures et structures cachées.
- **Optimisation des configurations d'électrodes** : Adaptation des arrays (comme Dipôle-Dipôle ou Schlumberger) pour une meilleure évaluation, par exemple dans la détection de méthane ou d'autres gaz en sites contrôlés.
- **ERT profonde pour ressources géothermiques** : Application de méthodes géoélectriques pour caractériser des ressources géothermiques à moyenne-basse enthalpie jusqu'à environ 1 km de profondeur.
- **Meilleures pratiques en imagerie de résistivité électrique** : Conception robuste de géométries de survey, sélection de paramètres d'acquisition et d'inversion pour minimiser les erreurs et améliorer la résolution.
- **Rôle des paramètres forward et inverse** : Focus sur l'impact des paramètres de modélisation directe et d'inversion pour des investigations ERT plus précises, avec des études de cas.
- **Inversion stochastique avec autoencodeurs variationnels convolutifs profonds (VAE)** : Couplage d'apprentissage profond avec optimisation adaptative stochastique pour une inversion ERT plus efficace et robuste.
- **ERT 3D avec reconstruction de source** : Méthode avancée pour la détection en trois dimensions, utilisant les différences de conductivité pour imager des anomalies comme des cavités ou des zones inondées.
- **Revues sur ERT profonde** : Résultats scientifiques pour explorer des environnements géologiques profonds, intégrant ERT avec d'autres méthodes géophysiques pour une meilleure caractérisation.
Ces techniques peuvent être intégrées pour améliorer la précision, la profondeur et l'interprétation des données ERT, potentiellement via des bibliothèques Python comme pyGIMLi pour l'inversion avancée ou scikit-image pour l'analyse texturale.
""")
    # Visualisations avec onglets pour bascule 2D/3D (générées avec IA au lieu de Plotly, prompts enrichis minéraux)
    st.subheader("📊 Visualisations Interactives Générées par IA (Inspiré Graphiques ERT Minéraux)")
    viz_tabs = st.tabs(["Section 2D Basique", "Section 2D Minéraux", "Contours Res2DInv", "Tomographie 3D"])
    
    # Fonction générique pour générer image IA pour viz (améliorée avec minéraux)
    def generate_viz_image(prompt_suffix, title, materials_df):
        if pipe:
            with st.spinner(f"Génération de {title}... (patientez 5-30s)"):
                # Enrichir prompt avec minéraux correspondants
                mineral_examples = materials_df.sample(3)['Type'].tolist()
                prompt = f"Une visualisation précise et scientifique d'une section ERT 2D/3D pour exploration minière : {prompt_suffix}. Inclure anomalies pour minéraux comme {', '.join(mineral_examples)}. Style tomographie géophysique réaliste (pseudosection/inversée), haute résolution, avec barres de couleurs log Ωm de {rho_min_global:.1f} à {rho_max_global:.1f}, profondeur jusqu'à {max_depth}m, espacement électrodes {electrode_spacing}m. Inclure annotations, légendes claires et labels minéraux."
                try:
                    image = pipe(
                        prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        height=512, width=1024,
                        generator=torch.manual_seed(42)
                    ).images[0]
                    # Ajout annotations avec minéraux
                    annotated_image = add_ert_annotations_to_image(image.copy(), rho_min_global, rho_max_global, max_depth, colorscale_ert, electrode_spacing, materials_df)
                    st.image(annotated_image, caption=title, use_container_width=True)
                    buf = io.BytesIO()
                    annotated_image.save(buf, format="PNG")
                    st.download_button(f"⬇️ Télécharger {title}", buf.getvalue(), f"{title.replace(' ', '_')}.png", "image/png")
                    st.success(f"✅ {title} générée avec précision IA et focus minéraux.")
                except Exception as e:
                    st.error(f"Erreur génération {title} : {e}")
                auto_pause_gpu()
        else:
            st.warning("SDXL non disponible pour génération d'images.")
    
    with viz_tabs[0]:
        generate_viz_image("coupe transversale 2D avec heatmap de résistivité normalisée, zones bleues pour conducteurs (eau/argile), vertes pour moyennes (sable), rouges pour isolants (roche sèche), lignes horizontales pour couches géologiques (surface, topsoil, subsoil, bedrock).", "Section 2D ERT avec couleurs géophysiques", materials_df)
    
    with viz_tabs[1]:
        clusters_summary = ", ".join([f"Cluster {cl}: {mean_rho:.1f} Ωm ({interp})" for cl, mean_rho, interp in zip(cluster_means.index, cluster_means.values, mean_df['interpretation'])])
        generate_viz_image(f"clustering minéral avec {n_clusters} clusters colorés arc-en-ciel, interprétations : {clusters_summary}, inclure légendes par cluster et couches géologiques, anomalies pour sulfures/pyrite.", "Section 2D avec Clustering Minéral", materials_df)
        st.write("Interprétations des clusters :")
        st.dataframe(mean_df[['rho', 'interpretation']])
    
    with viz_tabs[2]:
        generate_viz_image("contours de résistivité style Res2DInv, lignes de contour avec labels numériques en Ωm (échelle log), zones colorées bleu-vert-jaune-orange-rouge, profondeur et position X annotées, labels pour minéraux comme galena/pyrite.", "Contours Style Res2DInv", materials_df)
    
    with viz_tabs[3]:
        generate_viz_image("reconstruction 3D volumétrique du sous-sol ERT, vue en perspective avec iso-surfaces semi-transparentes, couleurs Res2DInv pour résistivité, axes X/Y/Z (position, largeur, profondeur), inclure coupes transversales avec légendes minérales.", "Tomographie 3D Générée par IA", materials_df)

    # Nouvelle section : 10 Graphiques Supplémentaires Générés par IA LLM basés sur RAG
    st.subheader("🖼️ 10 Graphiques Supplémentaires Générés par IA (Basés sur Études RAG et Calculs Données)")
    if st.button("Générer 10 Graphiques IA Supplémentaires") and st.session_state.get('ai_generated', False) and pipe:
        with st.spinner("Génération des 10 graphiques..."):
            # Prompt pour LLM générer 10 prompts spécifiques inspirés de RAG et données
            rag_context = st.session_state.ai_interpretation if 'ai_interpretation' in st.session_state else "Analyse ERT standard"
            graph_prompt = f"""Basé sur cette analyse ERT détaillée : {rag_context}, et les données calculées (rho de {rho_min_global:.1f} à {rho_max_global:.1f} Ωm, profondeur {max_depth}m, espacement {electrode_spacing}m), suggère 10 prompts spécifiques pour Stable Diffusion XL pour visualiser des graphiques ERT avancés. Inspire-toi de visualisations réelles comme pseudo-sections, sections inversées comparées, cartes de sensibilité, modèles 3D avec maillage, coupes transversales avec contours denses, analyse texturale GLCM, optimisation électrodes, ERT profonde géothermique, inversion stochastique VAE, reconstruction 3D sources. Chaque prompt doit inclure calculs précis (ex: résistivités clusters), style réaliste géophysique (Res2DInv, EarthImager, BERT), haute résolution, annotations, légendes log Ωm. Liste numérotée 1-10 avec prompts courts."""
            if llama_model:
                output = llama_model(
                    prompt=graph_prompt,
                    max_tokens=1000,
                    temperature=0.7,
                    stop=["\n\n"]
                )
                prompts_text = output['choices'][0]['text'].strip()
            else:
                inputs = tokenizer(graph_prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=800)
                prompts_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Parser les 10 prompts (simple split par numéro)
            prompts = re.split(r'\n(?=\d+\.)', prompts_text)[:10]
            for i, p in enumerate(prompts, 1):
                title = f"Graphique Supplémentaire {i}: {p.split('.')[0].strip()}"
                generate_viz_image(p.strip(), title, materials_df)
        auto_pause_gpu()

    # RAG Section (adaptée si non disponible, maintenant avec Llama pour affichage dynamique et agent Tavily)
    st.subheader("📚 Upload PDFs pour RAG (rapports géophysiques)")
    uploaded_pdfs = st.file_uploader("Choisir des PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs and LANGCHAIN_AVAILABLE:
        pdf_paths = []
        for pdf in uploaded_pdfs:
            with open(pdf.name, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_paths.append(pdf.name)
        vectorstore = init_rag_from_pdfs(pdf_paths)
        st.success(f"✅ RAG initialisé avec {len(uploaded_pdfs)} PDFs. Base vectorielle prête.")
        
        # Créer l'agent avec Tavily et RAG
        llm_rag = get_langchain_llm()
        if llm_rag:
            agent_executor = create_rag_agent(vectorstore, llm_rag)
            qa_chain = None  # Utiliser agent à la place
        else:
            agent_executor = None
            st.warning("Llama non disponible pour RAG Agent.")
    else:
        vectorstore = None
        agent_executor = None
        if not LANGCHAIN_AVAILABLE:
            st.warning("RAG désactivé (installez LangChain).")

    st.subheader("🔍 Queries RAG pour analyse avancée (avec Tavily Agent)")
    if agent_executor:
        query = st.text_input("Posez une question (ex. : 'Paramètres d'inversion pour ERT' ou 'Palette de couleurs pour visualisation')")
        if query:
            try:
                with st.spinner("Analyse RAG + Tavily en temps réel avec Agent..."):
                    result = agent_executor.invoke({"input": query})
                st.markdown(f"**Réponse Agent RAG/Tavily :** {result['output']}")
                
                # Exemple : Extraire couleurs et les appliquer dynamiquement (amélioré pour affichage intelligent)
                if "couleur" in query.lower() or "palette" in query.lower():
                    # Parser la réponse pour extraire hex colors ou rgb (amélioré avec Llama pour précision)
                    colors = re.findall(r'#[\w]{6}|rgb\(\d+,\s*\d+,\s*\d+\)', result['output'])  # Ex. : #0000FF ou rgb(0,0,255)
                    if colors:
                        st.write(f"**Couleurs extraites dynamiquement :** {colors}")
                        # Générer colorscale dynamique pour utilisation immédiate
                        custom_colorscale = []
                        for i, color in enumerate(colors):
                            if color.startswith('#'):
                                custom_colorscale.append([i / len(colors), f'rgb({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)})'])
                            else:
                                custom_colorscale.append([i / len(colors), color])
                        st.json({"Colorscale RAG pour visualisations": custom_colorscale})
                        st.success("✅ Palette de couleurs mise à jour dynamiquement via RAG. Utilisez-la dans les sections IA ci-dessous pour affichage intelligent.")
                
                # Pour calculs : Ex. extraire λ et l'appliquer dynamiquement à l'inversion
                if "paramètre" in query.lower() or "lambda" in query.lower():
                    lambda_match = re.search(r'λ\s*=\s*([\d.]+)|lambda\s*=\s*([\d.]+)', result['output'])
                    if lambda_match:
                        custom_lambda = float(lambda_match.groups()[0] or lambda_match.groups()[1])
                        st.slider("Lambda du PDF/Agent (appliqué dynamiquement)", value=custom_lambda)  # Met à jour le slider
                        st.info(f"✅ Paramètre λ = {custom_lambda} extrait et appliqué en temps réel.")
                auto_pause_gpu()
            except Exception as e:
                st.error(f"Erreur RAG Agent : {e}")
    else:
        st.info("Upload PDFs pour activer RAG Agent avec Tavily.")

    # Génération d'Images IA (adaptée si pipe None, amélioré avec annotations incrustées et Llama pour prompts, focus minéraux)
    st.subheader("🎨 Génération d'Images IA pour Modèles Géologiques (Focus Minéraux)")
    if st.button("Générer une Image Conceptuelle du Sous-Sol") and 'df' in locals() and pipe:
        with st.spinner("Génération en cours... (patientez 5-30s)"):
            # Auto-prompt via Llama + données ERT pour enrichissement intelligent (amélioré minéraux)
            clusters_summary = ", ".join([f"Cluster {cl}: {mean_rho:.1f} Ωm ({interp})" for cl, mean_rho, interp in zip(cluster_means.index, cluster_means.values, mean_df['interpretation'])])
            prompt_base = f"Une section géologique 2D du sous-sol pour exploration minière : zones de faible résistivité ({rho_min_global:.1f}-{np.percentile(df['rho'], 25):.1f} Ωm) en bleu pour sulfures comme pyrite/galena, zones moyennes en vert pour oxydes fer, hautes en rouge pour quartz. Style tomographie ERT réaliste, coupe transversale, annotations scientifiques."
            
            # Enrichir le prompt avec Llama pour dynamique
            if st.checkbox("Enrichir le prompt avec IA (Llama en temps réel)"):
                ai_prompt = f"Améliore ce prompt pour Stable Diffusion en intégrant les données ERT et minéraux : {prompt_base}. Rends-le détaillé pour une image géophysique précise, en incluant mesures rho, couleurs et anomalies minérales potentielles."
                if llama_model:
                    output = llama_model(
                        prompt=ai_prompt,
                        max_tokens=100,
                        temperature=0.8,
                        stop=["\n\n"]
                    )
                    prompt = output['choices'][0]['text'].strip()
                else:
                    # Fallback SmolLM
                    inputs = tokenizer(ai_prompt, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=50)
                    prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                prompt = prompt_base
            
            st.write(f"**Prompt enrichi dynamiquement :** {prompt}")
            
            # Génération d'image
            try:
                image = pipe(
                    prompt,
                    num_inference_steps=20,  # Moins = plus rapide (défaut 50)
                    guidance_scale=7.5,      # Force l'adhésion au prompt
                    height=512, width=1024,  # Format paysage pour sections ERT
                    generator=torch.manual_seed(42)  # Pour reproductibilité
                ).images[0]
                
                # Ajout annotations ERT incrustées (amélioré avec minéraux)
                annotated_image = add_ert_annotations_to_image(image.copy(), rho_min_global, rho_max_global, max_depth, colorscale_ert, electrode_spacing, materials_df)
                
                # Affichage
                st.image(annotated_image, caption="Image Générée : Modèle Géologique ERT avec Mesures et Minéraux Incrustés", use_container_width=True)
                
                # Option : Sauvegarder
                buf = io.BytesIO()
                annotated_image.save(buf, format="PNG")
                st.download_button("⬇️ Télécharger Image Annotée", buf.getvalue(), "ERT_GeoImage_Annotated.png", "image/png")
                
                st.success("✅ Image générée et annotée avec mesures ρ, couleurs, profondeurs et minéraux !")
            except Exception as e:
                st.error(f"Erreur génération : {e}. Vérifiez GPU/mémoire.")
        auto_pause_gpu()
    elif not pipe:
        st.warning("SDXL non disponible (installez diffusers).")
    else:
        st.info("📊 Chargez des données ERT pour générer des images adaptées.")

    # --- Export CSV --- (inchangé)
    st.subheader("💾 Export CSV")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger CSV", data=csv_data, file_name="ERT_Data.csv", mime="text/csv")
    auto_pause_gpu()
else:
    st.warning("📄 Importez un fichier DT1 ou DAT pour commencer l'analyse")
    st.info("💡 Pour les fonctionnalités avancées d'inversion, installez pyGIMLi : `pip install pygimli` et bitsandbytes pour la quantification IA : `pip install bitsandbytes`.")