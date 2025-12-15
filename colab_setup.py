"""
CLSO Colab Setup Helper

Quick installation script for Google Colab.
Handles all imports and path setup automatically.
"""

import sys
import os

def setup_clso():
    """Setup CLSO environment in Colab"""
    
    # Add repository to path
    repo_path = '/content/CLSO-training'
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    # Verify installation
    try:
        from src import BasisLibrary, CrystallineGPT2, GeneticOptimizer
        print("✅ CLSO modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying alternative import method...")
        
        # Alternative: direct imports
        try:
            from src.basis_library import BasisLibrary
            from src.crystalline_model import CrystallineGPT2
            from src.genetic_optimizer import GeneticOptimizer
            print("✅ CLSO modules imported successfully (alternative method)!")
            return True
        except ImportError as e2:
            print(f"❌ Alternative import also failed: {e2}")
            return False

def check_gpu():
    """Check GPU availability and specs"""
    import torch
    
    print("\n🔍 GPU Check:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️ No GPU detected. Training will be slow on CPU.")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("CLSO Setup for Google Colab")
    print("=" * 70)
    
    setup_success = setup_clso()
    gpu_available = check_gpu()
    
    print("\n" + "=" * 70)
    if setup_success and gpu_available:
        print("✅ Setup complete! You're ready to train CLSO on GPU!")
    elif setup_success:
        print("⚠️ Setup complete but no GPU detected.")
    else:
        print("❌ Setup failed. Please check error messages above.")
    print("=" * 70)
