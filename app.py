import streamlit as st
import pickle
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(page_title="Analisis Sentimen", page_icon="🎬", layout="centered")

st.title("🎬 Analisis Sentimen Review Film")
st.markdown("**IMDB Movie Reviews - NLP Classification**")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('count_vectorizer.pkl', 'rb') as f:
        bow = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return nb, lr, svm, bow, tfidf

try:
    nb_model, lr_model, svm_model, bow_vec, tfidf_vec = load_models()
    st.success("✅ Semua model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """Preprocessing teks: lowercase, remove special chars, tokenize, stem"""
    # Lowercase
    text = text.lower()
    # Remove URL, mention, hashtag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def safe_predict(model, vectorizer, text, model_name="Model"):
    """
    Prediksi dengan fallback untuk model yang tidak punya predict_proba
    """
    try:
        # Vectorize
        vec = vectorizer.transform([text])
        
        # Predict
        pred = model.predict(vec)[0]
        
        # Get confidence dengan fallback
        if hasattr(model, 'predict_proba'):
            # Model punya probability (Naive Bayes, Logistic Regression)
            proba = model.predict_proba(vec)[0]
            conf = float(max(proba))
        elif hasattr(model, 'decision_function'):
            # SVM tanpa probability - convert decision function
            decision = model.decision_function(vec)
            # Sigmoid untuk convert ke 0-1
            conf = float(1 / (1 + np.exp(-abs(decision[0]))))
        else:
            # Fallback default
            conf = 1.0
            
        return pred, conf
        
    except Exception as e:
        st.error(f"Error pada {model_name}: {str(e)}")
        return None, 0.0

# Input section
st.subheader("📝 Masukkan Review Film")
text_input = st.text_area(
    "Tulis review film di sini:",
    height=150,
    placeholder="Contoh: This movie is great! The acting was superb..."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

# Analysis
if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang menganalisis..."):
            # Preprocess
            processed = preprocess_text(text_input)
            
            # Debug (bisa dihapus)
            # st.write(f"Processed: {processed}")
            
            # Predictions dengan safe fallback
            nb_pred, nb_conf = safe_predict(nb_model, bow_vec, processed, "Naive Bayes")
            lr_pred, lr_conf = safe_predict(lr_model, tfidf_vec, processed, "Logistic Regression")
            svm_pred, svm_conf = safe_predict(svm_model, tfidf_vec, processed, "SVM")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")
        
        # Mapping label (0=negatif, 1=positif atau sebaliknya)
        label_map = {0: "Negatif 😠", 1: "Positif 😊"}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Naive Bayes",
                value=label_map.get(nb_pred, str(nb_pred)),
                delta=f"{nb_conf:.1%} confidence"
            )
            
        with col2:
            st.metric(
                label="Logistic Regression",
                value=label_map.get(lr_pred, str(lr_pred)),
                delta=f"{lr_conf:.1%} confidence"
            )
            
        with col3:
            st.metric(
                label="SVM",
                value=label_map.get(svm_pred, str(svm_pred)),
                delta=f"{svm_conf:.1%} confidence"
            )
        
        # Voting ensemble
        votes = [nb_pred, lr_pred, svm_pred]
        if None not in votes:
            final_pred = max(set(votes), key=votes.count)
            final_label = label_map.get(final_pred, str(final_pred))
            
            st.markdown("---")
            st.success(f"🏆 Prediksi Final (Voting): **{final_label}**")
            
            # Progress bar untuk confidence rata-rata
            avg_conf = (nb_conf + lr_conf + svm_conf) / 3
            st.progress(avg_conf)
            st.caption(f"Confidence rata-rata: {avg_conf:.1%}")
        else:
            st.error("❌ Terjadi error pada salah satu model")

# Footer
st.markdown("---")
st.caption("🎓 Tugas Besar NLP - STTC Candra Nuralim Fadilah | 2024")
