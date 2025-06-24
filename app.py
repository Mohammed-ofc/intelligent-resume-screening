import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt  # 📊 For the bar chart

from extract_text import extract_text_from_pdf
from generate_embeddings import model
from utils_enhanced import rank_resumes_hybrid  # ✅ Updated utils import

st.title("🧠 Intelligent Resume Screening System (Enhanced)")

jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
resumes = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

if jd_file and resumes:
    jd_text = jd_file.read().decode("utf-8")
    jd_embedding = model.encode(jd_text)

    resume_texts = []
    resume_names = []

    for uploaded_file in resumes:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text_from_pdf(temp_path)
        resume_texts.append(text)
        resume_names.append(uploaded_file.name)

    resume_embeddings = [model.encode(text) for text in resume_texts]
    ranked_resumes = rank_resumes_hybrid(jd_embedding, jd_text, resume_embeddings, resume_texts, resume_names, model)

    st.subheader("📊 Ranked Resumes:")
    for name, final_score, bert_score, skill_score in ranked_resumes:
        st.write(f"**{name}** — Final Score: {final_score:.2f} | BERT: {bert_score:.2f} | Skills: {skill_score:.2f}")

    # ✅ Add bar chart visualization
    st.subheader("📈 Final Match Score Chart")
    names = [x[0] for x in ranked_resumes]
    scores = [x[1] for x in ranked_resumes]

    fig, ax = plt.subplots()
    ax.barh(names, scores, color='mediumseagreen')
    ax.set_xlabel("Final Score (BERT + Skills)")
    ax.set_title("Resume vs Job Description Match")
    ax.invert_yaxis()
    st.pyplot(fig)




