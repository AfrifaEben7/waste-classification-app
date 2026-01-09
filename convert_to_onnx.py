"""
Convert PyTorch EfficientNet model to ONNX format
This script attempts to convert the .pth model to ONNX format
"""
import sys
import os

# Try to import torch with error handling
try:
    import torch
    import torch.nn as nn
    try:
        import timm
        USE_TIMM = True
    except ImportError:
        import torchvision.models as models
        USE_TIMM = False
        print("Warning: timm not found, using torchvision models")
    TORCH_AVAILABLE = True
    print("PyTorch imported successfully!")
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    print("\nFallback: Please run this script on a system where PyTorch works,")
    print("or provide the ONNX model file directly.")
    TORCH_AVAILABLE = False
    sys.exit(1)

def convert_model_to_onnx(pth_path, onnx_path, model_arch='efficientnet_b4', num_classes=6):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pth_path: Path to .pth model file
        onnx_path: Path to save .onnx model file
        model_arch: Model architecture name
        num_classes: Number of output classes
    """
    print(f"Loading model from {pth_path}...")
    
    # Create model architecture
    if 'efficientnet' in model_arch.lower():
        if 'b0' in model_arch.lower():
            img_size = 224
        elif 'b1' in model_arch.lower():
            img_size = 224
        elif 'b2' in model_arch.lower():
            img_size = 260
        elif 'b3' in model_arch.lower():
            img_size = 300
        elif 'b4' in model_arch.lower():
            img_size = 380
        elif 'b5' in model_arch.lower():
            img_size = 456
        elif 'b6' in model_arch.lower():
            img_size = 528
        elif 'b7' in model_arch.lower():
            img_size = 600
        else:
            img_size = 224
        
        # Try timm first (more likely for your model)
        if USE_TIMM:
            print("Using timm library for model creation...")
            model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
        else:
            print("Using torchvision library for model creation...")
            if 'b0' in model_arch.lower():
                model = models.efficientnet_b0(weights=None)
            elif 'b1' in model_arch.lower():
                model = models.efficientnet_b1(weights=None)
            elif 'b2' in model_arch.lower():
                model = models.efficientnet_b2(weights=None)
            elif 'b3' in model_arch.lower():
                model = models.efficientnet_b3(weights=None)
            elif 'b4' in model_arch.lower():
                model = models.efficientnet_b4(weights=None)
            elif 'b5' in model_arch.lower():
                model = models.efficientnet_b5(weights=None)
            elif 'b6' in model_arch.lower():
                model = models.efficientnet_b6(weights=None)
            elif 'b7' in model_arch.lower():
                model = models.efficientnet_b7(weights=None)
            else:
                model = models.efficientnet_b0(weights=None)
            
            # Modify classifier for torchvision models
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {model_arch}")
    
    # Load weights
    checkpoint = torch.load(pth_path, map_location='cpu')
    
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
    print("Model loaded successfully!")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to ONNX
    print(f"Exporting to ONNX format: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully converted to ONNX: {onnx_path}")
    print(f"Image size for this model: {img_size}x{img_size}")
    
    return img_size

if __name__ == '__main__':
    pth_model_path = 'models/efficientnet_b4_final.pth'
    onnx_model_path = 'models/efficientnet_b4_final.onnx'
    
    if not os.path.exists(pth_model_path):
        print(f"Error: Model file not found at {pth_model_path}")
        sys.exit(1)
    
    try:
        img_size = convert_model_to_onnx(
            pth_model_path,
            onnx_model_path,
            model_arch='efficientnet_b4',
            num_classes=6
        )
        print(f"\n✓ Conversion complete!")
        print(f"✓ ONNX model saved to: {onnx_model_path}")
        print(f"✓ Input image size: {img_size}x{img_size}")
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
