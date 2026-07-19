import os
from pypdf import PdfReader
import docx

def extract_text_from_pdf(file_path):
    """Extracts raw text from a PDF."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extracts raw text from a Word (.docx) file."""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text).strip()
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extracts raw text from a text (.txt) file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def load_documents_from_folder(folder_path):
    """It scans all the files inside a folder and extracts the text with live tracking."""
    extracted_data = []
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return extracted_data

    print(" -> Scanning directory tree for files...")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.split('.')[-1].lower()
            text = ""

            # Live trace to identify exactly which file is being read
            if ext in ['pdf', 'docx', 'txt']:
                print(f"    [Reading File] -> {file}")

            if ext == 'pdf':
                text = extract_text_from_pdf(file_path)
            elif ext == 'docx':
                text = extract_text_from_docx(file_path)
            elif ext == 'txt':
                text = extract_text_from_txt(file_path)
            
            if text:
                extracted_data.append({
                    "file_name": file,
                    "file_path": file_path,
                    "raw_text": text
                })
                
    return extracted_data

if __name__ == "__main__":
    test_folder = "./dataset" 
    documents = load_documents_from_folder(test_folder)
    print(f"Successfully extracted text from {len(documents)} documents.")