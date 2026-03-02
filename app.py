import streamlit as st
import pickle
import re

# Load models
@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('count_vectorizer.pkl', 'rb') as f:
        count_vec = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    return tfidf, count_vec, lr_model, nb_model, svm_model

tfidf, count_vec, lr_model, nb_model, svm_model = load_models()

# Simple stopwords
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
             'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
             'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
             'it', 'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him'}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# UI
st.set_page_config(page_title="Analisis Sentimen Film", page_icon="🎬")
st.title("🎬 Analisis Sentimen Ulasan Film")
st.markdown("Powered by Machine Learning")

text_input = st.text_area("Masukkan ulasan film (Bahasa Inggris):", height=150)

if st.button("🔍 Analisis Sentimen"):
    if text_input:
        with st.spinner("Sedang menganalisis..."):
            processed = clean_text(text_input)
            
            vec_tfidf = tfidf.transform([processed])
            vec_count = count_vec.transform([processed])
            
            lr_pred = int(lr_model.predict(vec_tfidf)[0])
            lr_conf = float(max(lr_model.predict_proba(vec_tfidf)[0]))
            
            nb_pred = int(nb_model.predict(vec_count)[0])
            nb_conf = float(max(nb_model.predict_proba(vec_count)[0]))
            
            svm_pred = int(svm_model.predict(vec_tfidf)[0])
            svm_conf = float(max(svm_model.predict_proba(vec_tfidf)[0]))
            
            st.success("Analisis selesai!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Logistic Regression", 
                         "Positif" if lr_pred == 1 else "Negatif",
                         f"{lr_conf:.2%}")
            
            with col2:
                st.metric("Naive Bayes",
                         "Positif" if nb_pred == 1 else "Negatif", 
                         f"{nb_conf:.2%}")
            
            with col3:
                st.metric("SVM",
                         "Positif" if svm_pred == 1 else "Negatif",
                         f"{svm_conf:.2%}")
            
            votes = [lr_pred, nb_pred, svm_pred]
            final = 1 if sum(votes) >= 2 else 0
            
            st.markdown("---")
            if final == 1:
                st.balloons()
                st.success(f"🎉 HASIL AKHIR: POSITIF ({sum(votes)}/3 model)")
            else:
                st.error(f"💔 HASIL AKHIR: NEGATIF ({3-sum(votes)}/3 model)")
    else:
        st.warning("Masukkan teks terlebih dahulu!")

st.markdown("---")
st.markdown("**Tugas Besar NLP**")
