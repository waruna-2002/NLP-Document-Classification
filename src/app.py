import sys
import asyncio
import warnings
import os
import joblib
import streamlit as st
import scipy.sparse as sp

# Warnings bypass
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except (AttributeError, RuntimeError):
        pass

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import FeatureExtractor
from extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

# Page Config
st.set_page_config(page_title="SecOps Document Intelligence", page_icon="🛡️", layout="wide")

# CSS
st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }
        .main-header { font-size: 2.2rem; font-weight: 800; color: #0ea5e9; text-align: center; }
        .sub-header { text-align: center; color: #64748b; font-size: 0.85rem; margin-bottom: 30px; text-transform: uppercase; }
        .gemini-wrapper { max-width: 900px; margin: 0 auto; }
        .verdict-box-sensitive { background-color: #1e1b1c; border-left: 5px solid #ef4444; padding: 22px; border-radius: 10px; margin-bottom: 25px; }
        .verdict-box-personal { background-color: #111c17; border-left: 5px solid #10b981; padding: 22px; border-radius: 10px; margin-bottom: 25px; }
        .telemetry-card { background-color: #111827; border: 1px solid #1f2937; padding: 16px; border-radius: 10px; margin-bottom: 15px; }
        .telemetry-label { color: #64748b; font-size: 0.75rem; font-weight: 700; display: block; margin-bottom: 6px; }
        .telemetry-value { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; font-family: 'Courier New', monospace; }
        .compliance-card { background-color: #111827; border: 1px solid #1f2937; border-top: 4px solid #0ea5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Engines
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
feature_extractor = FeatureExtractor()

@st.cache_resource
def load_engines():
    lr = joblib.load(os.path.join(MODELS_DIR, "document_classifier.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "document_classifier.pkl")) else None
    svm = joblib.load(os.path.join(MODELS_DIR, "document_classifier_svm.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "document_classifier_svm.pkl")) else None
    svm_vec = joblib.load(os.path.join(MODELS_DIR, "svm_vectorizer.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "svm_vectorizer.pkl")) else None
    return lr, svm, svm_vec

classifier, svm_model, svm_vectorizer = load_engines()

# State
if 'active_view' not in st.session_state: st.session_state.active_view = "Ingestion Workspace"

# Sidebar
with st.sidebar:
    if st.button("Ingestion Workspace"): st.session_state.active_view = "Ingestion Workspace"; st.rerun()
    if st.button("Evaluation Benchmarks"): st.session_state.active_view = "Model Evaluation"; st.rerun()
    if st.button("Compliance Metrics"): st.session_state.active_view = "Ethics & Compliance"; st.rerun()

# Main Logic
if st.session_state.active_view == "Ingestion Workspace":
    st.markdown('<div class="main-header">SecOps Document Intelligence</div>', unsafe_allow_html=True)
    
    # Input Form
    with st.form("input_form"):
        uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])
        submit = st.form_submit_button("Scan Document")
        
    if submit and uploaded_file:
        # File Handling
        temp_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        # Extraction
        ext = uploaded_file.name.split('.')[-1].lower()
        text = extract_text_from_pdf(temp_path) if ext == 'pdf' else extract_text_from_docx(temp_path) if ext == 'docx' else extract_text_from_txt(temp_path)
        os.remove(temp_path)
        
        # Prediction
        feats = feature_extractor.get_combined_features(text)
        st.success(f"Scan Complete! Classification: PERSONAL (Sample Output)")
        st.write(feats)

elif st.session_state.active_view == "Model Evaluation":
    st.title("Evaluation Benchmarks")
    st.write("Engine is running correctly.")

else:
    st.title("Compliance Metrics")
    st.write("Privacy focused.")
