from pathlib import Path
import torch
from safetensors.torch import load_file

# Chemin exact du fichier safetensors
MODEL_FILE = Path(r"C:\Users\Admin\.cache\huggingface\hub\models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt\snapshots\61c57447d7b0adc8a1a30b2b0adec7a8935aa2a3\model.safetensors")

# Charger les poids en tant que dictionnaire
state_dict = load_file(str(MODEL_FILE))
print("✅ Poids chargés")
print("Exemple d'une clé :", list(state_dict.keys())[:5])
print("Exemple de tensor :", state_dict[list(state_dict.keys())[0]].shape)
