# Converting PyTorch Model to ONNX

Since PyTorch has compatibility issues on your macOS ARM system, you need to convert your `.pth` model to ONNX format. Here are three options:

## Option 1: Use Google Colab (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Upload your `efficientnet_b4_final.pth` file
4. Run this code:

```python
!pip install torch torchvision onnx timm

import torch
import torch.nn as nn
import timm

# Create model architecture using timm (your model was trained with timm)
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=6)

# Load your weights
checkpoint = torch.load('efficientnet_b4_final.pth', map_location='cpu')

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Load the state dict
model.load_state_dict(state_dict)
model.eval()

print("✓ Model loaded successfully!")

# Export to ONNX with all data embedded (no external data file)
dummy_input = torch.randn(1, 3, 380, 380)  # B4 uses 380x380

# IMPORTANT: Use save_as_external_data=False to embed all weights in the .onnx file
torch.onnx.export(
    model,
    dummy_input,
    'efficientnet_b4_final.onnx',
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✓ Model converted successfully to ONNX!")

# Verify the ONNX model works
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('efficientnet_b4_final.onnx')
test_input = np.random.randn(1, 3, 380, 380).astype(np.float32)
output = session.run(None, {'input': test_input})
print(f"✓ ONNX model verified! Output shape: {output[0].shape}")
print(f"   Model has 6 classes output: {output[0].shape[1] == 6}")

# Check file size (should be ~65-70MB)
import os
file_size_mb = os.path.getsize('efficientnet_b4_final.onnx') / (1024*1024)
print(f"✓ ONNX file size: {file_size_mb:.1f} MB")

# Download the ONNX file
from google.colab import files
files.download('efficientnet_b4_final.onnx')
```

5. Download the generated `efficientnet_b4_final.onnx` file
6. Place it in `waste-classification-app/models/` directory

## Option 2: Use a Different Computer

If you have access to a Linux or Windows machine:

1. Copy `convert_to_onnx.py` and your `.pth` file to that machine
2. Install dependencies:
   ```bash
   pip install torch torchvision onnx
   ```
3. Run the conversion script:
   ```bash
   python convert_to_onnx.py
   ```
4. Copy the generated `.onnx` file back to your Mac

## Option 3: Docker on Mac

If you have Docker installed:

```bash
# Run PyTorch in a container
docker run -it --rm -v $(pwd):/workspace pytorch/pytorch:latest bash

# Inside the container
cd /workspace
pip install onnx
python convert_to_onnx.py
```

## After Conversion

Once you have `efficientnet_b4_final.onnx`:

1. Place it in the `models/` directory
2. The web application is already configured to use ONNX Runtime
3. Start the Flask app:
   ```bash
   conda activate waste-classifier
   python -m src.web.app
   ```

## Verifying the ONNX Model

After conversion, you can verify the model works:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/efficientnet_b4_final.onnx')
input_name = session.get_inputs()[0].name

# Create dummy input
dummy_input = np.random.randn(1, 3, 380, 380).astype(np.float32)

# Run inference
output = session.run(None, {input_name: dummy_input})
print(f"✓ Model works! Output shape: {output[0].shape}")
```

## Troubleshooting

- **Shape mismatch**: Ensure you use 380x380 for EfficientNet-B4
- **Import errors**: Make sure onnxruntime is installed: `pip install onnxruntime`
- **Model not loading**: Check that the .onnx file is in the correct directory

