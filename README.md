# 🧠 Intelligent Resume Screening System (NLP + BERT + Streamlit)

A smart application that ranks resumes based on their relevance to a given job description using **NLP**, **BERT embeddings**, and **skill matching** — similar to how human recruiters evaluate.

## 🚀 Features
- Upload a Job Description (TXT)
- Upload multiple Resumes (PDF)
- Extract text using PyMuPDF
- Generate semantic embeddings with Sentence Transformers
- Score resumes using **cosine similarity + skill matching**
- Ranks and visualizes resumes with a bar chart

## 🛠️ Tech Stack
- Python, Streamlit
- Sentence Transformers (BERT)
- PyMuPDF (for PDF parsing)
- Scikit-learn
- SpaCy (for skill extraction)
- Matplotlib (for visualization)

## 📂 Folder Structure
resume_screening/
├── app.py
├── extract_text.py
├── generate_embeddings.py
├── utils.py
├── utils_enhanced.py
├── sample_data/
│ ├── job_description.txt
│ └── resume1.pdf, ...
└── README.md


## 🙋‍♂️ Created By
**Mohammed Salman**  
📧 mohammed.salman.p.2004@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mohammed-salman-p-484a9431b/)
🌐 [GitHub](https://github.com/Mohammed-ofc)

## 📜 License
This project is licensed under the [MIT License](LICENSE).
