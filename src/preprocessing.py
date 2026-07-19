import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK data downloads (Runs only during the first execution)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')  # Added for newer NLTK versions
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')             # Added for newer NLTK versions
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Removes noisy characters, formatting errors, and extra spaces"""
        if not text:
            return ""
        # Lowercasing
        text = text.lower()
        # Remove URLs and Email addresses
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        # Remove punctuation and special characters (keeping alphanumerics and dashes for regex rules)
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        # Normalize multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text):
        """Runs the complete preprocessing pipeline on raw text"""
        # 1. Clean Text
        cleaned = self.clean_text(text)
        
        # 2. Tokenization
        tokens = word_tokenize(cleaned)
        
        # 3. Stop-words removal and Lemmatization
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        # Join back into a text string for ML vectorization compatibility
        return " ".join(processed_tokens)

# 💡 main.py එකට කෙලින්ම import කරලා ලෙහෙසියෙන් පාවිච්චි කරන්න හදපු Global Function එක
def clean_and_preprocess_text(text):
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)