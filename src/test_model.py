import os
import sys
import joblib

# Ensure custom extraction modules are accessible
from extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

# Adjust system path to import FeatureExtractor from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import FeatureExtractor

def predict_document_category(file_path):
    # Paths to the saved model check-points
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    model_path = os.path.join(model_dir, "document_classifier.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model assets not found at {model_path}. Run main.py first.")
        return

    # Load the rule-based intelligence framework
    classifier = joblib.load(model_path)
    feature_extractor = FeatureExtractor()
    
    # Extract file extension
    ext = file_path.split('.')[-1].lower()
    raw_text = ""
    
    # Extract text on the fly
    if ext == 'pdf':
        raw_text = extract_text_from_pdf(file_path)
    elif ext == 'docx':
        raw_text = extract_text_from_docx(file_path)
    elif ext in ['txt', 'log']:
        raw_text = extract_text_from_txt(file_path)
        
    if not raw_text.strip():
        print("Error: Could not extract readable text from the provided document.")
        return

    # Calculate custom dynamic security vector scores
    feats = feature_extractor.get_combined_features(raw_text)
    metadata_vector = [
        feats['has_nic'],
        feats['has_credit_card'],
        feats['has_phone'],
        feats['count_person'],
        feats['count_org'],
        feats['count_gpe']
    ]
    
    # Execute structural scoring matrix evaluation
    prediction = classifier.predict_single(raw_text, metadata_vector)
    
    print("\n" + "="*40)
    print(f"📄 Target File: {os.path.basename(file_path)}")
    print(f"🔒 Predicted Security Category: {prediction.upper()}")
    print("="*40)
    
    # Optional metadata flag summary log
    print(f" -> PII Indicators found - NIC: {feats['has_nic']}, Phone: {feats['has_phone']}, Cards: {feats['has_credit_card']}")

if __name__ == "__main__":
    # Test file path input logic
    print("Enter the absolute or relative path to the file you want to classify:")
    user_input_path = input("File Path: ").strip()
    
    # Strip quotes if user dragged and dropped the file into terminal
    user_input_path = user_input_path.strip("'\"")
    
    if os.path.exists(user_input_path):
        predict_document_category(user_input_path)
    else:
        print(f"Error: Target path '{user_input_path}' does not exist.")