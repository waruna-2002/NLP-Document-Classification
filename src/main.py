import os
import sys
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp

# Ensure local modules can be imported safely
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import preprocessing
from features import FeatureExtractor

# Dynamically resolve the best preprocessing function available
if hasattr(preprocessing, 'clean_and_preprocess_text'):
    preprocess_fn = preprocessing.clean_and_preprocess_text
elif hasattr(preprocessing, 'TextPreprocessor'):
    tp = preprocessing.TextPreprocessor()
    preprocess_fn = tp.preprocess
else:
    preprocess_fn = getattr(preprocessing, 'preprocess', getattr(preprocessing, 'clean_text', None))

class AdvancedDocumentClassifier:
    def __init__(self):
        # Using bigrams and limiting features to restrict overfitting spaces
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        # Applied stronger L2 regularization (C=0.1) to generalize text patterns 
        # and prevent heavy class biases on noisy or empty text states
        self.model = LogisticRegression(C=0.1, max_iter=1000, solver='liblinear')
        
    def train(self, texts, metadata_features, labels):
        print("\n💡 Vectorizing text and training advanced NLP model...")
        
        # 1. Clean all text streams through the NLP pipeline before training
        print("🔄 Running tokenization, stopwords removal and lemmatization...")
        processed_texts = []
        for idx, t in enumerate(texts):
            processed_texts.append(preprocess_fn(t) if preprocess_fn else t)
            if (idx + 1) % 500 == 0:
                print(f"   ⚙️ Cleaned {idx + 1} text streams...")
        
        # 2. Extract Text Features using TF-IDF
        print("📊 Extracting TF-IDF feature matrices...")
        X_text = self.vectorizer.fit_transform(processed_texts)
        
        # 3. Combine Text features with Rule-Based PII Metadata vectors
        X_combined = sp.hstack([X_text, sp.csr_matrix(metadata_features)], format='csr')
        
        # 4. Train the model
        print("🧠 Fitting Logistic Regression classifier...")
        self.model.fit(X_combined, labels)
        print("🎯 Training completed successfully.")

    def predict_single(self, text, metadata_vector):
        # Preprocess the raw text before prediction to match feature dimensions
        processed_text = preprocess_fn(text) if preprocess_fn else text
        
        # Generate the sparse text feature matrix
        X_text = self.vectorizer.transform([processed_text])
        
        # Featureless Document Intercept Check:
        # If the text vector contains zero vocabulary matches (e.g., non-English text strings)
        # AND all rule-based structural PII risk vectors are also zero, safely default 
        # to low-risk PERSONAL status instead of falling victim to model intercept bias.
        if X_text.nnz == 0 and sum(metadata_vector[:3]) == 0:
            return "PERSONAL"
            
        # Otherwise, combine features and let the trained ML model make the boundary decision
        X_combined = sp.hstack([X_text, sp.csr_matrix([metadata_vector])], format='csr')
        prediction = self.model.predict(X_combined)[0]
        
        # Streamlit front-end target tags match wenna predict karana output label eka
        # aniwaryayenma `.upper()` karala string ekak lesa return kirima.
        return str(prediction).upper()

    def save_model(self, model_path):
        joblib.dump(self, model_path)
        print(f"💾 Optimized Core Model saved at: {model_path}")

def run_pipeline(data_directory):
    print("🚀 Booting SOC-Ops Pipeline Training Stream...")
    texts = []
    metadata_features = []
    labels = []
    
    fe = FeatureExtractor()
    categories = ['company_sensitive', 'personal']
    
    file_count = 0  # Live progress counter
    
    for category in categories:
        cat_path = os.path.join(data_directory, category)
        if not os.path.exists(cat_path):
            print(f"⚠️ Warning: Folder path not found: {cat_path}")
            continue
            
        print(f"\n📂 Scanning files inside category: [{category.upper()}]...")
        for root, dirs, files in os.walk(cat_path):
            for file in files:
                if file.endswith(('.txt', '.pdf', '.docx')):
                    file_path = os.path.join(root, file)
                    
                    from extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
                    ext = file.split('.')[-1].lower()
                    
                    try:
                        if ext == 'pdf':
                            raw_text = extract_text_from_pdf(file_path)
                        elif ext == 'docx':
                            raw_text = extract_text_from_docx(file_path)
                        else:
                            raw_text = extract_text_from_txt(file_path)
                            
                        if not raw_text.strip():
                            continue
                            
                        # Extract Rule-based meta signals
                        feats = fe.get_combined_features(raw_text)
                        meta_vector = [
                            feats['has_nic'], feats['has_credit_card'], feats['has_phone'],
                            feats['count_person'], feats['count_org'], feats['count_gpe']
                        ]
                        
                        texts.append(raw_text)
                        metadata_features.append(meta_vector)
                        labels.append(category.upper())
                        
                        # Live Progress Update every 100 documents parsed
                        file_count += 1
                        if file_count % 100 == 0:
                            print(f"   ⏳ Ingested {file_count} documents into pipeline stream...")
                            
                    except Exception as e:
                        continue

    print(f"\n✅ Total documents successfully loaded into pipeline memory: {len(texts)}")

    if len(texts) == 0:
        print("❌ Error: Dataset directory is empty or unreadable.")
        return

    # Initialize and Train Model
    classifier = AdvancedDocumentClassifier()
    classifier.train(texts, metadata_features, labels)
    
    # Calculate training accuracy metrics for verification
    correct_predictions = 0
    print("\n📋 Evaluating final model performance on dataset...")
    for idx, (t, m, l) in enumerate(zip(texts, metadata_features, labels)):
        if classifier.predict_single(t, m) == l:
            correct_predictions += 1
        if (idx + 1) % 500 == 0:
            print(f"   🔎 Evaluated {idx + 1}/{len(texts)} entries...")
    
    system_acc = (correct_predictions / len(texts)) * 100
    print(f"\n📈 Final Engine Classification Accuracy: {system_acc:.2f}%")
    
    # Save the updated binary model assets
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "document_classifier.pkl")
    classifier.save_model(model_path)
    print("\n=== End-to-End Execution Completed Successfully ===")

if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
    run_pipeline(target_dir)