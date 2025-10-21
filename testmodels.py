# ============================================================
# 🧬 LifeModo AI Lab v2.0 – Streamlit All-in-One Multimodal
# Extraction PDF + OCR + Dataset Multimodal (Vision/Language/Audio) + Training + Test + Export
# ============================================================

import streamlit as st
import fitz, pytesseract, cv2, io, os, json, gc, shutil, time, zipfile
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
import torch
import torchaudio  # For audio processing
import speech_recognition as sr  # For speech-to-text
from sklearn.model_selection import train_test_split
from datasets import Dataset as HfDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import subprocess
import tensorflow as tf
import concurrent.futures
from functools import partial
import psutil  # For CPU monitoring
import GPUtil  # For GPU monitoring

# ============ CONFIGURATION ============

BASE_DIR = "lifemodo_data"
os.makedirs(BASE_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TEXT_DIR = os.path.join(BASE_DIR, "texts")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exported")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Configuration Tesseract pour Windows
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    st.warning(f"⚠️ Exécutable Tesseract non trouvé à {TESSERACT_CMD}. Veuillez installer Tesseract OCR et ajuster le chemin.")

st.set_page_config(page_title="LifeModo AI Lab Multimodal v2.0", layout="wide")
st.title("🧬 LifeModo AI Lab v2.0 – Créateur Multimodal IA : Vision, Langage, Audio")

# Gestion de l'état
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "r") as f:
        status = json.load(f)
else:
    status = {"processed_pdfs": []}
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

# Vérification GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device détecté : {device.upper()}")

# ============ UTILITAIRES ============

def log(msg):
    st.info(f"[{time.strftime('%H:%M:%S')}] {msg}")

def save_json(data, path):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    if device == "cuda":
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_load = gpu.load * 100
            gpu_mem = gpu.memoryUtil * 100
            return f"CPU: {cpu_percent}% | RAM: {mem_percent}% | GPU Load: {gpu_load}% | GPU Mem: {gpu_mem}%"
        else:
            return f"CPU: {cpu_percent}% | RAM: {mem_percent}% | No GPU detected"
    return f"CPU: {cpu_percent}% | RAM: {mem_percent}%"

# ============ EXTRACTION PDF ============

def extract_pdf(pdf_file):
    try:
        pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
        all_data = []
        for page_num, page in enumerate(pdf):
            text = page.get_text("text")
            text_file = os.path.join(TEXT_DIR, f"page_{page_num+1}.txt")
            with open(text_file, "w", encoding='utf-8') as f:
                f.write(text)

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                image_path = os.path.join(IMAGES_DIR, f"page_{page_num+1}_{img_index}.png")
                image.save(image_path)
                all_data.append({
                    "page": page_num+1,
                    "img_index": img_index,
                    "image_path": image_path,
                    "text_path": text_file
                })
        pdf.close()
        return all_data
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du PDF: {str(e)}")
        return []

# ============ OCR + ANNOTATIONS VISION ============

def ocr_and_annotate(image_path, class_id=0):
    try:
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise FileNotFoundError(f"Tesseract non trouvé à {pytesseract.pytesseract.tesseract_cmd}. Veuillez vérifier l'installation.")
        
        image = cv2.imread(image_path)
        if image is None:
            return None, None, []
        h, w, _ = image.shape
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        ocr_text = []
        annotations = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt: continue
            x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            annotations.append([class_id, cx, cy, bw_norm, bh_norm])
            ocr_text.append(txt)
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        annotated_path = image_path.replace(".png", "_annotated.png")
        cv2.imwrite(annotated_path, image)
        
        # Save YOLO labels with annotations
        label_file = image_path.replace(IMAGES_DIR, LABELS_DIR).replace(".png", ".txt")
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, "w", encoding='utf-8') as f:
            for ann in annotations:
                f.write(' '.join(map(str, ann)) + '\n')
        
        return " ".join(ocr_text), annotated_path, annotations
    except Exception as e:
        st.error(f"Erreur lors de l'OCR et annotation: {str(e)}")
        return None, None, []

# ============ TRAITEMENT AUDIO ============

def process_audio(audio_file):
    try:
        audio_path = os.path.join(AUDIO_DIR, audio_file.name)
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        
        # Speech-to-text using speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)  # Use Google API (requires internet)
        
        # Save transcript
        transcript_path = audio_path.replace(".wav", ".txt").replace(AUDIO_DIR, TEXT_DIR)
        with open(transcript_path, "w", encoding='utf-8') as f:
            f.write(transcript)
        
        # Load waveform for potential training
        waveform, sample_rate = torchaudio.load(audio_path)
        
        return {
            "audio_path": audio_path,
            "transcript": transcript,
            "waveform": waveform,
            "sample_rate": sample_rate
        }
    except Exception as e:
        st.error(f"Erreur lors du traitement audio: {str(e)}")
        return None

# ============ VISUALISATION DATASET ============

def visualize_dataset(dataset):
    if not dataset:
        st.warning("Dataset vide.")
        return
    df = pd.DataFrame(dataset)
    st.subheader("Tableau du Dataset")
    st.dataframe(df)
    
    st.subheader("Graphiques du Dataset")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count par type
    sns.countplot(data=df, x="type", ax=ax[0])
    ax[0].set_title("Distribution des Types")
    
    # Distribution des labels (si existants)
    if "label" in df.columns:
        sns.countplot(data=df, x="label", ax=ax[1])
        ax[1].set_title("Distribution des Labels")
    
    st.pyplot(fig)

# ============ GÉNÉRATION PROMPTS DYNAMIQUES ============

def generate_dynamic_prompts(train_data, prompt_template):
    prompts = []
    for d in train_data:
        text = d.get("text", "") + " " + d.get("ocr", "") + " " + d.get("transcript", "")
        prompt = prompt_template.format(text=text, label=d.get("label", "inconnu"))
        prompts.append(prompt)
    return prompts

# ============ DATASET CONSTRUCTION MULTIMODAL ============

def build_dataset(pdfs, audios=None, labels=None):
    dataset = []
    # Process PDFs with progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_pdfs = len(pdfs) if pdfs else 0
    for idx, pdf in enumerate(pdfs or []):
        pdf_name = pdf.name
        if pdf_name in status["processed_pdfs"]:
            log(f"{pdf_name} déjà traité. Passage au suivant.")
            continue
        log(f"Extraction du PDF : {pdf.name}")
        pages = extract_pdf(pdf)
        for item in pages:
            try:
                with open(item["text_path"], "r", encoding='utf-8') as f:
                    text_content = f.read()
                ocr_text, ann_image, annotations = ocr_and_annotate(item["image_path"])
                if ocr_text is None:
                    continue
                dataset.append({
                    "type": "vision",
                    "image": item["image_path"],
                    "annotated": ann_image,
                    "text": text_content,
                    "ocr": ocr_text,
                    "annotations": annotations,
                    "label": labels.get(item["image_path"], "texte") if labels else "texte"
                })
            except Exception as e:
                st.error(f"Erreur lors du traitement de la page {item['page']}: {str(e)}")
        status["processed_pdfs"].append(pdf_name)
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
        progress = (idx + 1) / total_pdfs
        progress_bar.progress(progress)
        progress_text.text(f"Extraction PDFs : {idx + 1}/{total_pdfs} ({progress*100:.1f}%)")
    
    # Process Audios
    for audio in audios or []:
        audio_data = process_audio(audio)
        if audio_data:
            dataset.append({
                "type": "audio",
                "audio_path": audio_data["audio_path"],
                "transcript": audio_data["transcript"],
                "waveform": audio_data["waveform"],
                "sample_rate": audio_data["sample_rate"],
                "label": labels.get(audio_data["audio_path"], "speech") if labels else "speech"
            })
    
    # Save dataset
    if dataset:
        dataset_path = os.path.join(BASE_DIR, "dataset.json")
        save_json(dataset, dataset_path)
        log(f"✅ Dataset multimodal enregistré : {dataset_path}")
    
    # Split dataset for training
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    progress_bar.progress(1.0)
    progress_text.text("Construction du dataset terminée !")
    return train_data, val_data

# ============ ENTRAÎNEMENT VISION (YOLO) ============

def train_vision_yolo(dataset_dir, epochs=50, imgsz=640, device=device):
    try:
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w", encoding='utf-8') as f:
            f.write(f"""
path: {dataset_dir}
train: images
val: images
nc: 1
names: ['texte']
""")
        
        weights_dir = os.path.join(MODEL_DIR, "vision_model/weights")
        last_checkpoint = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_checkpoint):
            model = YOLO(last_checkpoint)
            log("Checkpoint trouvé. Reprise de l'entraînement.")
        else:
            model = YOLO("yolov8n.pt")
            log("Aucun checkpoint trouvé. Démarrage depuis zéro.")
        
        # Barre de progression
        progress_bar = st.progress(0)
        progress_text = st.empty()
        monitor_text = st.empty()
        
        def on_train_epoch_end(trainer):
            progress = (trainer.epoch + 1) / epochs
            progress_bar.progress(progress)
            progress_text.text(f"Entraînement vision : Époque {trainer.epoch + 1}/{epochs} ({progress*100:.1f}%)")
            monitor_text.text(monitor_resources())
        
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, project=MODEL_DIR, name="vision_model", batch=16, resume=os.path.exists(last_checkpoint), device=device)
        best_model_path = os.path.join(MODEL_DIR, "vision_model/weights/best.pt")
        
        export_model_formats(best_model_path)
        
        progress_bar.progress(1.0)
        progress_text.text("Entraînement vision terminé !")
        
        return best_model_path
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement vision: {str(e)}")
        return None

# ============ EXPORT DES MODÈLES ============

def export_model_formats(model_path):
    try:
        model = YOLO(model_path)
        log("Export des modèles en cours...")
        
        model.export(format="onnx", path=os.path.join(EXPORT_DIR, "lifemodo.onnx"))
        model.export(format="saved_model", path=os.path.join(EXPORT_DIR, "lifemodo_tf"))
        
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(EXPORT_DIR, "lifemodo_tf"))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(os.path.join(EXPORT_DIR, "lifemodo.tflite"), "wb") as f:
            f.write(tflite_model)
        
        log("Conversion en TensorFlow.js...")
        tfjs_dir = os.path.join(EXPORT_DIR, "lifemodo_tfjs")
        if os.path.exists(tfjs_dir):
            shutil.rmtree(tfjs_dir)
        os.makedirs(tfjs_dir, exist_ok=True)
        subprocess.run(["tensorflowjs_converter", "--input_format=tf_saved_model", os.path.join(EXPORT_DIR, "lifemodo_tf"), tfjs_dir], check=True)
        
        log("=== Exports ONNX, TensorFlow, TFLite et TF.js terminés ===")
    except Exception as e:
        st.error(f"Erreur lors de l'exportation : {str(e)}")

# ============ ENTRAÎNEMENT LANGAGE (Transformers) ============

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_bar, progress_text, num_epochs, monitor_text):
        self.progress_bar = progress_bar
        self.progress_text = progress_text
        self.num_epochs = num_epochs
        self.monitor_text = monitor_text
    
    def on_epoch_end(self, args, state, control, **kwargs):
        progress = (state.epoch) / self.num_epochs
        self.progress_bar.progress(progress)
        self.progress_text.text(f"Entraînement langage : Époque {int(state.epoch)}/{self.num_epochs} ({progress*100:.1f}%)")
        self.monitor_text.text(monitor_resources())

def train_language(train_data, val_data, model_name="distilbert-base-uncased", epochs=3, dynamic_prompts=None, device=device):
    try:
        # Use dynamic prompts if provided
        if dynamic_prompts:
            texts = dynamic_prompts
        else:
            texts = [d["text"] + " " + d.get("ocr", "") + " " + d.get("transcript", "") for d in train_data]
        labels = [0 if "negative" in d["label"] else 1 for d in train_data]  # Dummy; adapt
        train_df = pd.DataFrame({"text": texts, "label": labels})
        val_texts = [d["text"] + " " + d.get("ocr", "") + " " + d.get("transcript", "") for d in val_data]
        val_labels = [0 if "negative" in d["label"] else 1 for d in val_data]
        val_df = pd.DataFrame({"text": val_texts, "label": val_labels})
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        
        train_dataset = HfDataset.from_pandas(train_df).map(tokenize_function, batched=True)
        val_dataset = HfDataset.from_pandas(val_df).map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        monitor_text = st.empty()
        
        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, "language_model"),
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
                **dict(zip(["precision", "recall", "f1"], precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average="binary")))
            }
        )
        
        trainer.add_callback(ProgressCallback(progress_bar, progress_text, epochs, monitor_text))
        
        trainer.train()
        best_model_path = os.path.join(MODEL_DIR, "language_model")
        trainer.save_model(best_model_path)
        
        progress_bar.progress(1.0)
        progress_text.text("Entraînement langage terminé !")
        
        log(f"✅ Modèle langage entraîné : {best_model_path}")
        return best_model_path
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement langage: {str(e)}")
        return None

# ============ ENTRAÎNEMENT AUDIO ============

def train_audio(train_data, val_data, epochs=10, device=device):
    try:
        audio_train = [d for d in train_data if d["type"] == "audio"]
        audio_val = [d for d in val_data if d["type"] == "audio"]
        if not audio_train:
            raise ValueError("Aucun données audio.")
        
        class AudioClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(16000, 2).to(device)
        
        model = AudioClassifier()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        monitor_text = st.empty()
        
        for epoch in range(epochs):
            for d in audio_train:
                waveform = d["waveform"].mean(dim=0)[:16000].to(device)
                label = torch.tensor([0 if "negative" in d["label"] else 1]).to(device)
                output = model(waveform.unsqueeze(0))
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            progress_text.text(f"Entraînement audio : {epoch + 1}/{epochs} ({progress*100:.1f}%)")
            monitor_text.text(monitor_resources())
        
        best_model_path = os.path.join(MODEL_DIR, "audio_model.pt")
        torch.save(model.state_dict(), best_model_path)
        
        progress_bar.progress(1.0)
        progress_text.text("Entraînement audio terminé !")
        
        log(f"✅ Modèle audio : {best_model_path}")
        return best_model_path
    except Exception as e:
        st.error(f"Erreur audio: {str(e)}")
        return None

# ============ TEST MULTIMODAL ============

def test_model(modality, file_path, model_path=None, text_model=None):
    st.subheader(f"🔍 Test {modality}")
    try:
        if modality == "vision":
            img = Image.open(file_path)
            st.image(img, caption="Image testée")
            if model_path:
                yolo = YOLO(model_path)
                results = yolo(img, device=device)
                st.image(results[0].plot(), caption="Détection YOLO")
        elif modality == "language":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            with open(file_path, "r", encoding='utf-8') as f:
                text = f.read()
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            st.write("🧠 Prédiction langage :", outputs.logits.argmax().item())
        elif modality == "audio":
            waveform, _ = torchaudio.load(file_path)
            model = torch.nn.Module()  # Load your model
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            output = model(waveform.mean(dim=0)[:16000].unsqueeze(0).to(device))
            st.write("🧠 Prédiction audio :", output.argmax().item())
        if text_model:
            res = text_model(file_path)
            st.write("🧠 NLP :", res[0]['generated_text'])
    except Exception as e:
        st.error(f"Erreur test: {str(e)}")

# ============ INTERFACE STREAMLIT ============

st.sidebar.title("⚙️ Contrôle Multimodal v2.0")
mode = st.sidebar.radio("Choisir le mode :", ["📥 Importation Données", "🧠 Entraînement IA", "🧪 Test du Modèle", "📤 Export Dataset/Modèles"])
preview_images = st.sidebar.checkbox("Prévisualisation images", value=False)

if mode == "📥 Importation Données":
    st.header("📥 Importer PDF/Audio pour dataset multimodal")
    uploaded_pdfs = st.file_uploader("PDFs :", type=["pdf"], accept_multiple_files=True)
    uploaded_audios = st.file_uploader("Audios :", type=["wav", "mp3"], accept_multiple_files=True)
    custom_labels = st.text_input("Labels JSON: {'file_path': 'label'}", "{}")
    try:
        labels = json.loads(custom_labels)
    except:
        labels = {}
        st.warning("Labels invalide.")
    if uploaded_pdfs or uploaded_audios:
        train_data, val_data = build_dataset(uploaded_pdfs, uploaded_audios, labels)
        dataset = train_data + val_data
        st.success(f"{len(dataset)} échantillons (Train: {len(train_data)}, Val: {len(val_data)}).")
        visualize_dataset(dataset)
        if preview_images and st.checkbox("Aperçu"):
            for d in train_data[:5]:
                if d["type"] == "vision":
                    st.image(d["annotated"], caption=d["ocr"])
                    st.text_area("Texte :", d["text"], height=150)
                elif d["type"] == "audio":
                    st.audio(d["audio_path"])
                    st.text_area("Transcript :", d["transcript"], height=150)

elif mode == "🧠 Entraînement IA":
    st.header("🧠 Entraîner IA multimodaux")
    modalities = st.multiselect("Modèles :", ["Vision (YOLO)", "Langage (Transformers)", "Audio (Torchaudio)"])
    epochs = st.slider("Époques :", 1, 50, 10)
    prompt_template = st.text_input("Template prompt langage (ex: 'Classifie {text} comme {label}')", "")
    if st.button("🚀 Lancer entraînement"):
        dataset_path = os.path.join(BASE_DIR, "dataset.json")
        if not os.path.exists(dataset_path):
            st.error("Dataset non trouvé.")
        else:
            with open(dataset_path, "r", encoding='utf-8') as f:
                dataset = json.load(f)
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
            dynamic_prompts = generate_dynamic_prompts(train_data, prompt_template) if prompt_template else None
            
            def train_mod(mod):
                if mod == "Vision (YOLO)":
                    return train_vision_yolo(BASE_DIR, epochs)
                elif mod == "Langage (Transformers)":
                    return train_language(train_data, val_data, epochs=epochs, dynamic_prompts=dynamic_prompts)
                elif mod == "Audio (Torchaudio)":
                    return train_audio(train_data, val_data, epochs)
            
            if len(modalities) > 1 and device == "cuda":
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(partial(train_mod, mod)) for mod in modalities]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
            else:
                for mod in modalities:
                    train_mod(mod)

elif mode == "🧪 Test du Modèle":
    st.header("🧪 Tester IA")
    modality = st.selectbox("Modality :", ["Vision", "Language", "Audio"])
    file_uploader_type = {"Vision": ["png", "jpg"], "Language": ["txt"], "Audio": ["wav", "mp3"]}
    file = st.file_uploader(f"Fichier {modality} :", type=file_uploader_type.get(modality, []))
    model_type = st.selectbox("Modèle supp. :", ["Aucun", "Image-to-Text", "Text-Generation"])
    if file:
        file_path = os.path.join(BASE_DIR, f"test.{file.name.split('.')[-1]}")
        with open(file_path, "wb") as f:
            f.write(file.read())
        model_path = os.path.join(MODEL_DIR, f"{modality.lower()}_model/weights/best.pt" if modality == "Vision" else f"{modality.lower()}_model")
        if os.path.exists(model_path):
            text_model = None
            if model_type == "Image-to-Text":
                text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
            elif model_type == "Text-Generation":
                text_model = pipeline("text-generation", model="gpt2")
            test_model(modality.lower(), file_path, model_path, text_model)
        else:
            st.error(f"⚠️ Modèle {modality} non trouvé.")

elif mode == "📤 Export Dataset/Modèles":
    st.header("📤 Exporter")
    if st.button("Exporter ZIP"):
        zip_path = os.path.join(BASE_DIR, "export.zip")
        zip_directory(BASE_DIR, zip_path)
        with open(zip_path, "rb") as f:
            st.download_button("Télécharger ZIP", f, file_name="lifemodo_export.zip")
        log("✅ Export prêt.")