"""
Verification script to check if everything is set up correctly.
Run this before running main.py or evaluation.py
"""

import os
import sys

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version: {}.{}.{} (OK)".format(version.major, version.minor, version.micro))
        return True
    else:
        print("✗ Python version: {}.{}.{} (Need 3.8+)".format(version.major, version.minor, version.micro))
        return False

def check_packages():
    """Check if required packages are installed"""
    required = [
        'langchain',
        'langchain_community', 
        'chromadb',
        'sentence_transformers',
        'ollama',
        'rouge_score',
        'nltk',
        'sklearn'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_ollama():
    """Check if Ollama is running and Mistral is available"""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="mistral")
        response = llm.invoke("Say 'OK' if you're working")
        print("✓ Ollama server is running")
        print("✓ Mistral model is available")
        return True
    except Exception as e:
        print("✗ Ollama connection failed")
        print(f"  Error: {e}")
        print("\n  Fix:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull Mistral: ollama pull mistral")
        return False

def check_corpus():
    """Check if corpus files exist"""
    corpus_dir = "./corpus"
    required_files = [
        "speech1.txt", "speech2.txt", "speech3.txt",
        "speech4.txt", "speech5.txt", "speech6.txt"
    ]
    
    if not os.path.exists(corpus_dir):
        print(f"✗ Corpus directory not found: {corpus_dir}")
        print("  Run: python setup_corpus.py")
        return False
    
    missing = []
    for file in required_files:
        filepath = os.path.join(corpus_dir, file)
        if os.path.exists(filepath):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            missing.append(file)
    
    if missing:
        print("\n⚠️  Run: python setup_corpus.py")
        return False
    return True

def check_test_dataset():
    """Check if test dataset exists"""
    if os.path.exists("test_dataset.json"):
        print("✓ test_dataset.json exists")
        return True
    else:
        print("✗ test_dataset.json not found")
        print("  Please ensure test_dataset.json is in the project root")
        return False

def main():
    print("\n" + "="*60)
    print("  AMBEDKARGPT - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks = []
    
    print("1. Checking Python version...")
    checks.append(check_python_version())
    print()
    
    print("2. Checking Python packages...")
    checks.append(check_packages())
    print()
    
    print("3. Checking Ollama and Mistral...")
    checks.append(check_ollama())
    print()
    
    print("4. Checking corpus files...")
    checks.append(check_corpus())
    print()
    
    print("5. Checking test dataset...")
    checks.append(check_test_dataset())
    print()
    
    print("="*60)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run:")
        print("  • python main.py          - Interactive Q&A")
        print("  • python evaluation.py    - Full evaluation")
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        print("\nQuick fixes:")
        print("  • Packages: pip install -r requirements.txt")
        print("  • Corpus: python setup_corpus.py")
        print("  • Ollama: ollama serve (in separate terminal)")
        print("  • Mistral: ollama pull mistral")
    print()

if __name__ == "__main__":
    main()