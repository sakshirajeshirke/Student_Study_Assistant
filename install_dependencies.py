import subprocess
import sys

def install_packages():
    print("Installing required packages...")
    packages = [
        "streamlit",
        "pdfplumber",
        "qdrant-client",
        "torch",
        "pandas",
        "sentence-transformers",
        "rank-bm25",
        "langchain-ollama",
        "langchain",
        "python-docx",  # For the docx module
        "ydata-profiling",
        "reportlab",
        "matplotlib",
        "seaborn",
        "numpy",
        "transformers",
        "Pillow",  # PIL/Image
        "pytesseract",
        "nltk",
        "wordcloud",
        "youtube-dl"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Download NLTK data
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages()