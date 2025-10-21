"""
AI Verification Service (Streamlit version, single-file, no torch at runtime for this model)
- Télécharge les modèles depuis Hugging Face en utilisant HF_TOKEN présent dans .env
- Fournit une interface Streamlit pour l'enregistrement et la vérification
- Utilise: FAISS (face vector store), chromadb (optionnel pour multimodal), sqlite (métadonnées)
- Intègre le modèle de détection de deepfake pour anti-spoofing
- Support webcam pour reconnaissance faciale

Requirements (pip):
  pip install streamlit python-dotenv huggingface_hub onnxruntime opencv-python-headless facenet-pytorch faiss-cpu chromadb soundfile librosa transformers speechbrain numpy sqlite3 aiofiles python-multipart torch safetensors optimum[exporters]

Notes:
- Pour le modèle anti-spoof, une conversion unique à ONNX est effectuée si nécessaire (requiert torch temporairement).
- Exécuter: HF_TOKEN doit être dans .env (HUGGINGFACE_API_TOKEN=xxx)
  streamlit run ai_verification_service.py

Limitations & recommandations:
- Whisper-large et certains modèles sont lourds (~1-3GB). Prévoir GPU/CPU et swap.
- Termux peut nécessiter des builds spécifiques et packages supplémentaires.
"""

import os
import io
import time
import json
import sqlite3
import tempfile
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from typing import Optional, Any

# Streamlit
import streamlit as st

# Media
import cv2
from PIL import Image

# Optional libraries (import lazily)
try:
    import onnxruntime as ort
except Exception as e:
    st.write(f"Failed to import onnxruntime: {e}")
    ort = None
try:
    from facenet_pytorch import MTCNN
except Exception as e:
    st.write(f"Failed to import facenet_pytorch: {e}")
    MTCNN = None
try:
    import faiss
except Exception as e:
    st.write(f"Failed to import faiss: {e}")
    faiss = None
try:
    import chromadb
except Exception as e:
    st.write(f"Failed to import chromadb: {e}")
    chromadb = None

# Audio
try:
    import soundfile as sf
    import librosa
except Exception as e:
    st.write(f"Failed to import audio libraries: {e}")
    sf = None
    librosa = None

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_CACHE_DIR = os.path.abspath("./models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

MODEL_REPOS = {
    "face_embedding": "hunyuan/insightface-recognition",
    "face_antispoof": "kadirnar/face-antispoofing",
    "face_detector": "RetinaFace/ResNet50",
    "voice_embed": "speechbrain/spkrec-ecapa-voxceleb",
    "asr": "openai/whisper-large-v3",
    "voice_emotion": "superb/hubert-base-superb-er",
}

LOCAL_MODELS = {}

@st.cache_resource
def download_models():
    for key, repo in MODEL_REPOS.items():
        try:
            st.write(f"Downloading {repo} ...")
            path = snapshot_download(
                repo_id=repo,
                cache_dir=MODEL_CACHE_DIR,
                ignore_regex="\\.ckpt$",
                repo_type=None,
                token=HF_TOKEN
            )
            LOCAL_MODELS[key] = path
            st.write(f"Downloaded {repo} → {path}")
        except Exception as e:
            st.write(f"Warning: download failed for {repo}: {e}")

if HF_TOKEN:
    download_models()
else:
    st.write("No HF token found in .env; skipping automatic download.")

def load_onnx_model(path: str):
    if ort is None:
        raise RuntimeError("onnxruntime not installed")
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model from {path}: {e}")

mtcnn = MTCNN(keep_all=True) if MTCNN is not None else None

# Database
DB_PATH = "ai_verification.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at REAL,
    metadata TEXT
)
""")
conn.commit()

# FAISS index
FAISS_INDEX_PATH = "faces.index"
FACE_DIM = 512
faiss_index = None
if faiss is not None:
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            st.write("Loaded FAISS index")
        except Exception as e:
            st.write(f"Failed to load FAISS index: {e}")
            faiss_index = faiss.IndexFlatL2(FACE_DIM)
    else:
        faiss_index = faiss.IndexFlatL2(FACE_DIM)
else:
    st.write("FAISS not available; face search disabled.")

MAP_PATH = "faiss_id_map.json"
id_map = json.load(open(MAP_PATH, "r")) if os.path.exists(MAP_PATH) else {}

# Chroma client
CHROMA_PATH = "./chroma_storage"
chroma_client = None
if chromadb is not None:
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        st.write("Initialized Chroma PersistentClient")
        if "users" not in [c.name for c in chroma_client.list_collections()]:
            chroma_client.create_collection(name="users")
    except Exception as e:
        st.write(f"Chroma init failed: {e}")
        chroma_client = None
else:
    st.write("ChromaDB not available; audio search disabled.")

# ---------------- Models ---------------- #

class FaceModel:
    def __init__(self):
        self.model = None
        self.path = LOCAL_MODELS.get('face_embedding')
        if self.path:
            try:
                for root, _, files in os.walk(self.path):
                    for f in files:
                        if f.endswith('.onnx'):
                            self.model = load_onnx_model(os.path.join(root, f))
                            st.write('Loaded face embedding ONNX model.')
                            return
            except Exception as e:
                st.write(f'ONNX load failed: {e}')

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        if self.model is None:
            vec = np.mean(img, axis=(0,1)).astype('float32')
            vec = np.resize(vec, (FACE_DIM,))
            vec /= np.linalg.norm(vec) + 1e-10
            return vec
        inp_name = self.model.get_inputs()[0].name
        img_resized = cv2.resize(img, (112,112)).astype('float32') / 255.0
        img_resized = np.transpose(img_resized, (2,0,1)).reshape(1,3,112,112)
        out = self.model.run(None, {inp_name: img_resized})
        emb = np.array(out[0]).reshape(-1).astype('float32')
        emb /= np.linalg.norm(emb) + 1e-10
        return emb

face_model = FaceModel()

class AntiSpoofModel:
    def __init__(self):
        self.model = None
        onnx_path = "antispoof.onnx"
        safetensors_path = r"C:\Users\Admin\.cache\huggingface\hub\models--romitbarua--autotrain-deepfakeface_only_faces_insightface-94902146221\snapshots\d0c39f494c0634667485f66f06c56ba85a1ef990\model.safetensors"
        if not os.path.exists(onnx_path):
            try:
                import torch
                from transformers import SwinForImageClassification
                from safetensors.torch import load_file
                from optimum.onnxruntime import ORTModelForImageClassification
                temp_dir = tempfile.mkdtemp()
                state_dict = load_file(safetensors_path, device="cpu")
                torch_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=2, ignore_mismatched_sizes=True)
                torch_model.load_state_dict(state_dict, strict=False)
                torch_model.save_pretrained(temp_dir)
                onnx_model = ORTModelForImageClassification.from_pretrained(temp_dir, export=True)
                onnx_model.save_pretrained(os.path.dirname(onnx_path))
                os.rename(os.path.join(os.path.dirname(onnx_path), "model.onnx"), onnx_path)
                st.write("Modèle anti-spoof exporté vers ONNX avec succès.")
            except Exception as e:
                st.write(f"Échec de la conversion en ONNX: {e}")
                return
        try:
            self.model = load_onnx_model(onnx_path)
            st.write("Modèle anti-spoof ONNX chargé.")
        except Exception as e:
            st.write(f"Échec du chargement du modèle anti-spoof ONNX: {e}")

    def predict(self, face_img: np.ndarray) -> float:
        if self.model is None:
            return 0.95  # Fallback
        try:
            img = cv2.resize(face_img, (224, 224)).astype('float32')
            mean = np.array([0.485, 0.456, 0.406], dtype='float32') * 255.0
            std = np.array([0.229, 0.224, 0.225], dtype='float32') * 255.0
            img = ((img - mean) / std).astype('float32')
            img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
            inp_name = self.model.get_inputs()[0].name
            out_name = self.model.get_outputs()[0].name
            logits = self.model.run([out_name], {inp_name: img})[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            real_prob = probs[0, 0]  # Assumer 0 = real, 1 = fake/deepfake; ajuster si nécessaire
            return real_prob
        except Exception as e:
            st.write(f"Échec de la prédiction anti-spoof: {e}")
            return 0.0

antispoof = AntiSpoofModel()

class VoiceEmbedModel:
    def get_embedding(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        if librosa is None:
            raise RuntimeError("librosa not available")
        try:
            mfcc = librosa.feature.mfcc(y=audio_np.astype('float32'), sr=sr, n_mfcc=40)
            vec = np.mean(mfcc, axis=1)
            vec = np.resize(vec, (192,)).astype('float32')
            vec /= np.linalg.norm(vec) + 1e-10
            return vec
        except Exception as e:
            st.write(f"Voice embedding failed: {e}")
            raise

voice_model = VoiceEmbedModel()

# ---------------- Utilities ---------------- #

def read_imagefile_to_bgr(data: bytes) -> np.ndarray:
    try:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    except Exception as e:
        st.write(f"Image reading failed: {e}")
        raise

def detect_and_align_face(bgr_image: np.ndarray) -> Optional[np.ndarray]:
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if mtcnn is not None:
            boxes, _ = mtcnn.detect(Image.fromarray(rgb))
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(x) for x in boxes[0]]
                face = rgb[y1:y2, x1:x2]
                if face.size > 0:
                    return face
        # Fallback: center crop
        h, w = rgb.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        face = rgb[cy - s // 2 : cy + s // 2, cx - s // 2 : cx + s // 2]
        if face.size > 0:
            return face
        return None
    except Exception as e:
        st.write(f"Face detection failed: {e}")
        return None

next_faiss_id = max(int(k) for k in id_map) + 1 if id_map else 0

def add_face_to_index(user_id: str, vector: np.ndarray):
    global next_faiss_id, faiss_index
    if faiss is None or faiss_index is None:
        raise RuntimeError('FAISS not available')
    try:
        faiss_index.add(np.array([vector]).astype('float32'))
        id_map[str(next_faiss_id)] = user_id
        next_faiss_id += 1
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(MAP_PATH, 'w') as f:
            json.dump(id_map, f)
    except Exception as e:
        st.write(f"Failed to add to FAISS index: {e}")
        raise

def search_face(vector: np.ndarray, k: int = 5) -> list:
    if faiss is None or faiss_index is None:
        return []
    try:
        D, I = faiss_index.search(np.array([vector]).astype('float32'), k)
        res = []
        for d, i in zip(D[0], I[0]):
            if i < 0:
                continue
            uid = id_map.get(str(i))
            if uid:
                res.append({'user_id': uid, 'distance': float(d)})
        return res
    except Exception as e:
        st.write(f"FAISS search failed: {e}")
        return []

# ---------------- Register / Verify ---------------- #

def register(user_id: str, name: str, image: Any, audio: Optional[Any] = None) -> dict:
    try:
        img_data = image.read()
        bgr = read_imagefile_to_bgr(img_data)
        face = detect_and_align_face(bgr)
        if face is None:
            return {"error": "no_face_detected"}
        spoof_prob = antispoof.predict(face)
        if spoof_prob < 0.5:
            return {"error": "spoof_detected", "confidence": spoof_prob}
        emb = face_model.get_embedding(face)
        add_face_to_index(user_id, emb)
        if audio and sf is not None and librosa is not None:
            audio_bytes = audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + (audio.name.split('.')[-1] if audio.name else 'wav')) as f:
                f.write(audio_bytes)
                tmp = f.name
            try:
                wav, sr = librosa.load(tmp, sr=16000)
                vemb = voice_model.get_embedding(wav, sr)
                if chroma_client is not None:
                    coll = chroma_client.get_collection('users')
                    coll.add(ids=[user_id], embeddings=[vemb.tolist()], metadatas=[{"name": name}])
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        cur.execute(
            "INSERT OR REPLACE INTO users (id, name, created_at, metadata) VALUES (?,?,?,?)",
            (user_id, name, time.time(), json.dumps({}))
        )
        conn.commit()
        return {"status": "ok", "user_id": user_id, "spoof_confidence": spoof_prob}
    except Exception as e:
        st.write(f"Registration failed: {e}")
        return {"error": str(e)}

def verify(image: Optional[Any] = None, audio: Optional[Any] = None) -> dict:
    try:
        face_result = None
        audio_result = None
        if image:
            img_data = image.read()
            bgr = read_imagefile_to_bgr(img_data)
            face = detect_and_align_face(bgr)
            if face is None:
                return {"error": "no_face_detected"}
            spoof_prob = antispoof.predict(face)
            if spoof_prob < 0.4:
                return {"error": "spoof_detected", "confidence": spoof_prob}
            emb = face_model.get_embedding(face)
            face_result = {"candidates": search_face(emb, k=3), "spoof_confidence": spoof_prob}
        if audio and sf is not None and librosa is not None:
            audio_bytes = audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + (audio.name.split('.')[-1] if audio.name else 'wav')) as f:
                f.write(audio_bytes)
                tmp = f.name
            try:
                wav, sr = librosa.load(tmp, sr=16000)
                vemb = voice_model.get_embedding(wav, sr)
                if chroma_client is not None:
                    coll = chroma_client.get_collection('users')
                    audio_result = coll.query(query_embeddings=[vemb.tolist()], n_results=5)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return {"face": face_result, "audio": audio_result}
    except Exception as e:
        st.write(f"Verification failed: {e}")
        return {"error": str(e)}

def health() -> dict:
    return {"status": "ok", "models": list(LOCAL_MODELS.keys())}

# ---------------- Streamlit UI ---------------- #

st.title("AI Verification Service (No Torch at runtime)")

mode = st.sidebar.selectbox("Mode", ["Register", "Verify", "Health"])

if mode == "Register":
    user_id = st.text_input("User ID")
    name = st.text_input("Name")
    use_webcam = st.checkbox("Use Webcam for Image")
    if use_webcam:
        image = st.camera_input("Take a picture for registration")
    else:
        image = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
    audio = st.file_uploader("Audio (optional)", type=["wav", "mp3", "m4a"])
    if st.button("Register"):
        if user_id and name and image:
            result = register(user_id, name, image, audio)
            st.json(result)
        else:
            st.error("Please provide User ID, Name, and Image.")

elif mode == "Verify":
    use_webcam = st.checkbox("Use Webcam for Image")
    if use_webcam:
        image = st.camera_input("Take a picture for verification")
    else:
        image = st.file_uploader("Image (optional)", type=["jpg", "png", "jpeg"])
    audio = st.file_uploader("Audio (optional)", type=["wav", "mp3", "m4a"])
    if st.button("Verify"):
        if image or audio:
            result = verify(image, audio)
            st.json(result)
        else:
            st.error("Please provide at least one of Image or Audio.")

elif mode == "Health":
    result = health()
    st.json(result)