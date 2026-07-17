import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentVectorizer:
    def __init__(self, max_features=5000):
        # Initialize the TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, raw_documents):
        """Fits the vectorizer on the corpus and returns the TF-IDF feature matrix"""
        if not raw_documents:
            return None
        return self.vectorizer.fit_transform(raw_documents)

    def transform(self, new_documents):
        """Transforms new documents based on the already fitted vocabulary"""
        if not new_documents:
            return None
        return self.vectorizer.transform(new_documents)

    def save_vectorizer(self, filepath):
        """Saves the trained vectorizer instance using joblib for deployment inference"""
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer successfully saved to {filepath}")

    def load_vectorizer(self, filepath):
        """Loads a previously saved vectorizer instance"""
        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer successfully loaded from {filepath}")

if __name__ == "__main__":
    # Test script block with a few preprocessed sample texts
    sample_corpus = [
        "quick brown fox jumping lazy dog report sent",
        "secret project configuration file warning",
        "system log network access database connection established"
    ]
    
    vec = DocumentVectorizer(max_features=100)
    tfidf_matrix = vec.fit_transform(sample_corpus)
    
    print("--- Vectorization Test Output ---")
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Extracted Vocabulary Features: {vec.vectorizer.get_feature_names_out()}")