"""
Optimized Vitis AI Quantization Script for Waste Classification
Target: KRIA KV260 (DPU Acceleration)
Run inside Vitis AI Docker: conda activate vitis-ai-pytorch
"""

import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import sys
import json
from pathlib import Path

# ----------------------------
# Model Definition (MUST EXACTLY match training)
# ----------------------------
class WasteMobileNetV2(nn.Module):
    """MobileNetV2 for 6-class waste classification with INT8 quantization support"""
    def __init__(self, num_classes=6, dropout=0.2):
        super().__init__()
        from torchvision import models
        
        # Load base MobileNetV2
        base_model = models.mobilenet_v2(weights=None)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier as Sequential to match training
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    'model_path': 'best_model.pth',  # Your trained model
    'calib_dir': 'calibration_dataset',  # 508 validation images for calibration
    'output_dir': 'quantize_result',
    'batch_size': 1,
    'num_calib_batches': 508,  # Use all validation images
    'img_size': 224,
    'num_classes': 6,  # cardboard, glass, metal, paper, plastic, trash
    'dropout': 0.2
}

def validate_setup():
    """Validate all required files exist"""
    print("="*70)
    print("VALIDATING SETUP")
    print("="*70)
    
    issues = []
    
    # Check model file
    if not os.path.exists(CONFIG['model_path']):
        issues.append(f"Model file not found: {CONFIG['model_path']}")
        issues.append("  → Expected: best_model.pth")
    else:
        size_mb = os.path.getsize(CONFIG['model_path']) / (1024*1024)
        print(f"✓ Model file: {CONFIG['model_path']} ({size_mb:.2f} MB)")
    
    # Check calibration directory
    if not os.path.exists(CONFIG['calib_dir']):
        issues.append(f"Calibration directory not found: {CONFIG['calib_dir']}")
        issues.append("  → Expected structure: WasteDataset/val/[cardboard,glass,metal,paper,plastic,trash]/")
    else:
        # Check for class folders
        expected_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        class_folders = [f for f in Path(CONFIG['calib_dir']).iterdir() if f.is_dir()]
        
        if len(class_folders) != 6:
            issues.append(f"Expected 6 class folders in {CONFIG['calib_dir']}, found {len(class_folders)}")
        else:
            print(f"✓ Calibration directory: {CONFIG['calib_dir']}")
            total_images = 0
            for cf in sorted(class_folders):
                img_count = len(list(cf.glob('*.jpg'))) + len(list(cf.glob('*.png')))
                total_images += img_count
                print(f"  - {cf.name:12s}: {img_count:3d} images")
            print(f"  Total: {total_images} images")
            
            if total_images < 400:
                issues.append(f"Warning: Only {total_images} calibration images (recommended: 500+)")
    
    if issues:
        print("\n❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    
    print("\n✓ All validation checks passed\n")

def load_model():
    """Load trained model"""
    print("="*70)
    print("LOADING MODEL")
    print("="*70)
    
    device = torch.device("cpu")  # Quantization runs on CPU
    model = WasteMobileNetV2(
        num_classes=CONFIG['num_classes'],
        dropout=CONFIG['dropout']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {CONFIG['model_path']}")
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    
    # Handle different checkpoint formats
    state_dict = None
    if isinstance(checkpoint, dict):
        # Try different keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  Format: {'model_state_dict': ...}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("  Format: {'state_dict': ...}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  Format: {'model': ...}")
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
            print("  Format: Direct state dict")
    else:
        # Direct state dict (ordered dict)
        state_dict = checkpoint
        print("  Format: Direct state dict (OrderedDict)")
    
    # Debug: Print first few keys
    print("\n  First 5 keys in state_dict:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"    {key}")
    
    # Try to load
    try:
        model.load_state_dict(state_dict, strict=True)
        print("\n✓ Model loaded successfully (strict=True)")
    except Exception as e:
        print(f"\n⚠️  Strict loading failed: {e}")
        print("\n  Attempting flexible loading...")
        
        # Try loading with strict=False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"\n  Missing keys ({len(missing)}):")
            for key in missing[:10]:  # Show first 10
                print(f"    - {key}")
        
        if unexpected:
            print(f"\n  Unexpected keys ({len(unexpected)}):")
            for key in unexpected[:10]:  # Show first 10
                print(f"    - {key}")
        
        # Check if critical layers loaded
        if 'features' in str(missing) or 'classifier' in str(missing):
            print("\n❌ Critical layers missing! Cannot proceed.")
            sys.exit(1)
        else:
            print("\n✓ Model loaded with warnings (non-critical mismatches)")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (approx): {total_params * 4 / (1024*1024):.2f} MB (FP32)")
    
    return model, device

def prepare_calibration_data():
    """Prepare calibration dataset"""
    print("\n" + "="*70)
    print("PREPARING CALIBRATION DATA")
    print("="*70)
    
    # CRITICAL: Use EXACT same preprocessing as training
    # Standard ImageNet preprocessing for MobileNetV2
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    calib_dataset = ImageFolder(CONFIG['calib_dir'], transform=transform)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,  # Don't shuffle for reproducibility
        num_workers=4,
        pin_memory=False  # CPU quantization
    )
    
    print(f"✓ Calibration dataset loaded")
    print(f"  Total images: {len(calib_dataset)}")
    print(f"  Classes: {calib_dataset.classes}")
    print(f"  Class to index mapping:")
    for cls, idx in sorted(calib_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"    {idx}: {cls}")
    print(f"  Batches to use: {min(CONFIG['num_calib_batches'], len(calib_loader))}")
    
    return calib_dataset, calib_loader

def run_calibration(model, calib_loader, device):
    """Run calibration phase"""
    print("\n" + "="*70)
    print("CALIBRATION PHASE")
    print("="*70)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Create dummy input for quantizer initialization
    dummy_input = torch.randn(CONFIG['batch_size'], 3, CONFIG['img_size'], CONFIG['img_size'])
    
    # Initialize quantizer in calibration mode
    print("\nInitializing quantizer (calib mode)...")
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=model,
        input_args=dummy_input,
        output_dir=CONFIG['output_dir'],
        device=device
    )
    
    quantized_model = quantizer.quant_model
    quantized_model.eval()
    
    # Run calibration
    print(f"\nRunning calibration on {min(CONFIG['num_calib_batches'], len(calib_loader))} batches...")
    print("This will take a few minutes...\n")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(calib_loader):
            if idx >= CONFIG['num_calib_batches']:
                break
            
            images = images.to(device)
            _ = quantized_model(images)
            
            if (idx + 1) % 50 == 0:
                progress = (idx + 1) / min(CONFIG['num_calib_batches'], len(calib_loader)) * 100
                print(f"  Progress: {idx + 1}/{min(CONFIG['num_calib_batches'], len(calib_loader))} batches ({progress:.1f}%)")
    
    print(f"\n✓ Calibration complete ({idx + 1} batches processed)")
    
    # Export calibration config
    quantizer.export_quant_config()
    print("✓ Quantization config exported")
    
    return quantizer

def run_test_quantization(model, calib_loader, device):
    """Test quantized model accuracy"""
    print("\n" + "="*70)
    print("TEST PHASE - Measuring INT8 Quantization Accuracy")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(CONFIG['batch_size'], 3, CONFIG['img_size'], CONFIG['img_size'])
    
    # Initialize quantizer in test mode
    print("\nInitializing quantizer (test mode)...")
    quantizer = torch_quantizer(
        quant_mode='test',
        module=model,
        input_args=dummy_input,
        output_dir=CONFIG['output_dir'],
        device=device
    )
    
    quantized_model = quantizer.quant_model
    quantized_model.eval()
    
    # Test accuracy
    print("\nTesting quantized model on validation set...")
    correct = 0
    total = 0
    class_correct = [0] * 6
    class_total = [0] * 6
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    with torch.no_grad():
        for images, labels in calib_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = quantized_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    overall_acc = 100 * correct / total
    
    print(f"\n{'='*70}")
    print("INT8 QUANTIZATION ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"\nOverall INT8 Accuracy: {overall_acc:.2f}%")
    print(f"  Correct: {correct}/{total}")
    
    # Show per-class accuracy
    print(f"\nPer-Class INT8 Accuracy:")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {name:12s}: {acc:5.2f}% ({class_correct[i]:3d}/{class_total[i]:3d})")
        else:
            print(f"  {name:12s}: No samples")
    
    # Expected accuracy comparison
    print(f"\nQuantization Analysis:")
    print(f"  Expected FP32 accuracy: ~92-95%")
    print(f"  Achieved INT8 accuracy: {overall_acc:.2f}%")
    
    if overall_acc >= 90.0:
        print(f"  ✓ Excellent! INT8 accuracy ≥ 90%")
    elif overall_acc >= 85.0:
        print(f"  ✓ Good quantization (85-90%)")
    else:
        print(f"  ⚠️  WARNING: Accuracy below 85%")
        print(f"  Consider:")
        print(f"    - Verifying preprocessing matches training exactly")
        print(f"    - Using more diverse calibration images")
        print(f"    - Checking if model loaded correctly")
    
    return quantizer, overall_acc

def export_xmodel(quantizer):
    """Export to XMODEL format for DPU"""
    print("\n" + "="*70)
    print("EXPORTING XMODEL FOR KV260 DPU")
    print("="*70)
    
    try:
        quantizer.export_xmodel(
            deploy_check=False,
            output_dir=CONFIG['output_dir']
        )
        
        print("\n✓ XMODEL export successful!")
        
        # Find generated xmodel file
        xmodel_files = list(Path(CONFIG['output_dir']).glob('*.xmodel'))
        
        if xmodel_files:
            print(f"\nGenerated files in {CONFIG['output_dir']}:")
            for xmodel in xmodel_files:
                size_mb = xmodel.stat().st_size / (1024*1024)
                print(f"  ✓ {xmodel.name} ({size_mb:.2f} MB)")
                
                # Rename to expected name if needed
                expected_name = "WasteMobileNetV2_int.xmodel"
                if xmodel.name != expected_name:
                    new_path = xmodel.parent / expected_name
                    xmodel.rename(new_path)
                    print(f"  → Renamed to: {expected_name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ XMODEL export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_quantization_report(quant_acc, calib_dataset):
    """Save detailed quantization report"""
    report = {
        'project': 'Waste Classification on KRIA KV260',
        'model_architecture': 'MobileNetV2',
        'num_classes': 6,
        'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        'input_size': f"{CONFIG['img_size']}x{CONFIG['img_size']}x3",
        'quantization': {
            'format': 'INT8',
            'calibration_images': len(calib_dataset),
            'calibration_batches': CONFIG['num_calib_batches'],
            'accuracy': f"{quant_acc:.2f}%"
        },
        'output_files': {
            'xmodel': 'quantize_result/WasteMobileNetV2_int.xmodel',
            'config': 'quantize_result/quant_info.json'
        },
        'next_steps': [
            'Compile for KV260 using vai_c_xir',
            'Deploy to KRIA KV260 board',
            'Run inference with VART runtime'
        ]
    }
    
    report_path = Path(CONFIG['output_dir']) / 'quantization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Detailed report saved: {report_path}")

def print_next_steps(quant_acc):
    """Print KV260 deployment instructions"""
    print("\n" + "="*70)
    print("NEXT STEPS - KV260 DEPLOYMENT")
    print("="*70)
    
    xmodel_files = list(Path(CONFIG['output_dir']).glob('*.xmodel'))
    xmodel_name = xmodel_files[0].name if xmodel_files else "WasteMobileNetV2_int.xmodel"
    
    print(f"""
📋 STEP 1: COMPILE FOR KV260 DPU
   
   vai_c_xir \\
     -x {CONFIG['output_dir']}/{xmodel_name} \\
     -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \\
     -o compiled_model \\
     -n waste_classifier

📋 STEP 2: VERIFY COMPILATION
   
   ls -lh compiled_model/waste_classifier.xmodel
   # Expected size: ~2-3 MB

📋 STEP 3: TRANSFER TO KV260
   
   scp compiled_model/waste_classifier.xmodel ubuntu@<kv260-ip>:~/
   scp inference.py ubuntu@<kv260-ip>:~/
   scp -r test_images/ ubuntu@<kv260-ip>:~/

📋 STEP 4: RUN INFERENCE ON KV260
   
   # On KV260 board
   python3 inference.py test_images/glass/glass103.jpg

📊 EXPECTED PERFORMANCE:
   - INT8 Accuracy: ~{quant_acc:.1f}% (±1%)
   - Model Size: ~2.4 MB
   - Inference: Real-time on KV260 DPU
   - Classes: cardboard, glass, metal, paper, plastic, trash

⚠️  IMPORTANT NOTES:
   - The compiled model is HYBRID (CPU + DPU)
   - DPU expects feature tensor [16, 56, 56, 24]
   - Preprocessing must run on CPU first
   - See README.md for architecture details
""")
    
    print("="*70)

def main():
    """Main quantization pipeline"""
    print("\n" + "="*70)
    print("VITIS AI QUANTIZATION - WASTE CLASSIFICATION")
    print("Target: KRIA KV260 Vision AI Starter Kit")
    print("="*70)
    
    try:
        # Step 1: Validate setup
        validate_setup()
        
        # Step 2: Load model
        model, device = load_model()
        
        # Step 3: Prepare calibration data
        calib_dataset, calib_loader = prepare_calibration_data()
        
        # Step 4: Run calibration
        run_calibration(model, calib_loader, device)
        
        # Step 5: Test quantization
        quantizer, quant_acc = run_test_quantization(model, calib_loader, device)
        
        # Step 6: Export XMODEL
        success = export_xmodel(quantizer)
        
        if not success:
            print("\n❌ Quantization pipeline failed!")
            sys.exit(1)
        
        # Step 7: Save report
        save_quantization_report(quant_acc, calib_dataset)
        
        # Step 8: Print next steps
        print_next_steps(quant_acc)
        
        print("\n" + "="*70)
        print("✅ QUANTIZATION COMPLETE!")
        print(f"INT8 Accuracy: {quant_acc:.2f}%")
        print(f"Output: {CONFIG['output_dir']}/WasteMobileNetV2_int.xmodel")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()