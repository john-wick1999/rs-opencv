import torch

print(f"Setup complete. Using torch tro{torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")