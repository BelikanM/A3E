import pynvml
pynvml.nvmlInit()
print("GPU détecté :", pynvml.nvmlDeviceGetCount())
pynvml.nvmlShutdown()
