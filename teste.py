import sys
from pathlib import Path
import torch

# Chemin vers ton modèle
MODEL_PATH = Path(r"C:\Users\Admin\.cache\huggingface\hub\models--dylanebert--LGM-full\snapshots\d8db5110f68eebab2dfb7687691babd146a933cd")

# Ajouter le chemin aux modules personnalisés
sys.path.append(str(MODEL_PATH))

# Importer les classes du modèle
try:
    from lgm.lgm import LGMModel
    from image_encoder.model import ImageEncoder
except ImportError as e:
    print("Erreur d'importation. Vérifie les chemins et les fichiers du modèle.")
    raise e

# Vérifier si CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# Charger l'Image Encoder
image_encoder_path = MODEL_PATH / "image_encoder" / "model.safetensors"
image_encoder = ImageEncoder()
state_dict = torch.load(image_encoder_path, map_location=device)
image_encoder.load_state_dict(state_dict)
image_encoder.to(device)
image_encoder.eval()
print("Image Encoder chargé avec succès !")

# Charger le modèle LGM
lgm_model_path = MODEL_PATH / "lgm" / "diffusion_pytorch_model.safetensors"
lgm_model = LGMModel()
state_dict_lgm = torch.load(lgm_model_path, map_location=device)
lgm_model.load_state_dict(state_dict_lgm)
lgm_model.to(device)
lgm_model.eval()
print("Modèle LGM chargé avec succès !")

# Test rapide : passage d'un tenseur dummy
dummy_image = torch.randn(1, 3, 256, 256).to(device)  # exemple image RGB 256x256
with torch.no_grad():
    features = image_encoder(dummy_image)
    output = lgm_model(features)

print("Test effectué avec succès. Shape output :", output.shape)
