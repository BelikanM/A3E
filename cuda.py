import torch
print("PyTorch version:", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("Version CUDA :", torch.version.cuda)
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")
