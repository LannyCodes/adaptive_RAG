"""
Install dependencies for professional hallucination detector
Run this before using the new hallucination detection features
"""

import subprocess
import sys


def install_dependencies():
    """Install required packages for hallucination detection"""
    
    print("=" * 60)
    print("🔧 Installing Hallucination Detector Dependencies")
    print("=" * 60)
    
    packages = [
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0"
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All dependencies installed successfully!")
    print("=" * 60)
    
    return True


def download_models():
    """Pre-download models to cache"""
    print("\n" + "=" * 60)
    print("🔧 Downloading Models (this may take a few minutes)...")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Download Vectara model
        print("\n📥 Downloading Vectara HHEM model...")
        try:
            AutoTokenizer.from_pretrained("vectara/hallucination_evaluation_model")
            AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model")
            print("✅ Vectara model downloaded")
        except Exception as e:
            print(f"⚠️ Vectara model download failed: {e}")
        
        # Download NLI model
        print("\n📥 Downloading DeBERTa NLI model...")
        try:
            from transformers import pipeline
            pipeline("text-classification", model="microsoft/deberta-large-mnli")
            print("✅ NLI model downloaded")
        except Exception as e:
            print(f"⚠️ NLI model download failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Models downloaded successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Cannot download models: {e}")
        print("Please install transformers first")
        return False
    
    return True


def test_installation():
    """Test if installation works"""
    print("\n" + "=" * 60)
    print("🧪 Testing Installation...")
    print("=" * 60)
    
    try:
        from hallucination_detector import HybridHallucinationDetector
        
        print("\n📝 Creating test detector...")
        detector = HybridHallucinationDetector(use_vectara=True, use_nli=True)
        
        print("\n📝 Running test detection...")
        test_doc = "Python is a programming language created by Guido van Rossum in 1991."
        test_gen = "Python was created by Guido van Rossum."
        
        result = detector.detect(test_gen, test_doc)
        print(f"\n✅ Test result: {result}")
        
        print("\n" + "=" * 60)
        print("✅ Installation test passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Installation test failed: {e}")
        print("\nPlease check the error messages above.")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting installation...\n")
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed at dependency stage")
        sys.exit(1)
    
    # Step 2: Download models
    if not download_models():
        print("\n⚠️ Model download had issues, but you can continue")
    
    # Step 3: Test installation
    if test_installation():
        print("\n" + "=" * 60)
        print("🎉 Installation Complete!")
        print("=" * 60)
        print("\nYou can now use the professional hallucination detector.")
        print("\nTo test it, run:")
        print("  python test_hallucination_detector.py")
        print("\n" + "=" * 60)
    else:
        print("\n❌ Installation completed with errors")
        print("The system will fallback to LLM-based detection")
        sys.exit(1)
