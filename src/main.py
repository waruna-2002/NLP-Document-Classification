import os
from extraction import extract_text_from_pdf, extract_text_from_docx
from preprocessing import TextPreprocessor
# Importing FeatureExtractor from root as it resides outside src folder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import FeatureExtractor
from vectorizer import DocumentVectorizer

def run_pipeline(data_directory):
    print("=== Starting NLP Document Classification Pipeline ===")
    
    # Initialize all subsystems
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    vectorizer = DocumentVectorizer(max_features=100)
    
    raw_texts = []
    metadata_features = []
    file_names = []
    
    # 1. Text Extraction Stage
    if not os.path.exists(data_directory):
        print(f"Error: Directory '{data_directory}' does not exist.")
        return
        
    for file in os.listdir(data_directory):
        file_path = os.path.join(data_directory, file)
        text = ""
        
        if file.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
        if text.strip():
            file_names.append(file)
            raw_texts.append(text)
            print(f"[Extracted] {file}")

    if not raw_texts:
        print("No documents found to process.")
        return

    print(f"\nSuccessfully extracted text from {len(raw_texts)} documents.")

    # 2. NLP Preprocessing & Feature Engineering Stage
    processed_texts = []
    print("\n--- Processing Text & Engineering Features ---")
    for doc_text in raw_texts:
        # Get cleaned text
        cleaned = preprocessor.preprocess(doc_text)
        processed_texts.append(cleaned)
        
        # Extract custom structural features (Regex + NER)
        feats = feature_extractor.get_combined_features(doc_text)
        metadata_features.append(feats)

    # 3. Vectorization Stage
    print("\n--- Generating TF-IDF Representations ---")
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Display summary results
    print("\n=== Pipeline Execution Summary ===")
    for idx, name in enumerate(file_names):
        print(f"\nDocument: {name}")
        print(f"  Processed Text: {processed_texts[idx][:60]}...")
        print(f"  Structural Features: {metadata_features[idx]}")
        
    print(f"\nFinal TF-IDF Feature Matrix Shape: {tfidf_matrix.shape}")
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Pointing to the local dataset directory for testing
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
    run_pipeline(target_dir)