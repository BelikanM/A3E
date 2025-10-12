import torch

print("==== Vérification PyTorch + CUDA ====")

# Version PyTorch
print(f"Version PyTorch installée : {torch.__version__}")

# Vérifier si GPU disponible
gpu_available = torch.cuda.is_available()
print(f"GPU disponible : {gpu_available}")

if gpu_available:
    print(f"Nom GPU : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPUs : {torch.cuda.device_count()}")
    print(f"Version CUDA compilée dans PyTorch : {torch.version.cuda}")
    print(f"Version cuDNN : {torch.backends.cudnn.version()}")
else:
    print("CUDA non disponible. Vérifie ton driver GPU et l’installation CUDA.")

# Test rapide tensor sur GPU
if gpu_available:
    try:
        x = torch.randn(3, 3).cuda()
        print("Test tensor sur GPU : OK")
    except Exception as e:
        print(f"Erreur lors du test tensor GPU : {e}")
