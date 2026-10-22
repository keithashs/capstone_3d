import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything

print("=" * 60)
print("MapAnything model test")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading MapAnything...")
model = MapAnything.from_pretrained("facebook/map-anything")
model = model.to(device)
model.eval()

print("\nSUCCESS")
print("MapAnything loaded onto:", device)

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"GPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved:  {reserved:.2f} GB")