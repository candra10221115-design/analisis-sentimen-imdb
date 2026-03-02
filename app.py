import streamlit as st
import pickle
import numpy as np
import re
import string

# Page config
st.set_page_config(page_title="Analisis Sentimen", page_icon="🎬", layout="centered")

st.title("🎬 Analisis Sentimen Review Film")
st.markdown("**IMDB Movie Reviews - UAS CANDRA NLP Classification**")
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

# Simple preprocessing tanpa NLTK
def preprocess_text(text):
    """Simple preprocessing tanpa NLTK"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def safe_predict(model, vectorizer, text, model_name="Model"):
    """Prediksi dengan fallback"""
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            conf = float(max(proba))
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(vec)
            conf = float(1 / (1 + np.exp(-abs(decision[0]))))
        else:
            conf = 1.0
            
        return pred, conf
        
    except Exception as e:
        st.error(f"Error pada {model_name}: {str(e)}")
        return None, 0.0

# UI
st.subheader("📝 Masukkan Review Film")
text_input = st.text_area("Tulis review film di sini:", height=150, 
                          placeholder="Contoh: This movie is great!")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Silakan masukkan teks!")
    else:
        with st.spinner("Sedang menganalisis..."):
            processed = preprocess_text(text_input)
            
            nb_pred, nb_conf = safe_predict(nb_model, bow_vec, processed, "Naive Bayes")
            lr_pred, lr_conf = safe_predict(lr_model, tfidf_vec, processed, "Logistic Regression")
            svm_pred, svm_conf = safe_predict(svm_model, tfidf_vec, processed, "SVM")
        
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")
        
        label_map = {0: "Negatif 😠", 1: "Positif 😊"}
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Naive Bayes", label_map.get(nb_pred, str(nb_pred)), f"{nb_conf:.1%}")
        with col2:
            st.metric("Logistic Regression", label_map.get(lr_pred, str(lr_pred)), f"{lr_conf:.1%}")
        with col3:
            st.metric("SVM", label_map.get(svm_pred, str(svm_pred)), f"{svm_conf:.1%}")
        
        votes = [nb_pred, lr_pred, svm_pred]
        if None not in votes:
            final_pred = max(set(votes), key=votes.count)
            st.success(f"🏆 Final: **{label_map.get(final_pred, str(final_pred))}**")

st.markdown("---")
st.caption("🎓 UAS-NLP-Candra-Nuralim-Fadilah | 2026")
