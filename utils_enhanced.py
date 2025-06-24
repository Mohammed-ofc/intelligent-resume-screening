# utils_enhanced.py
import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import util

nlp = spacy.load("en_core_web_sm")

# Define known skill keywords (expandable)
SKILL_KEYWORDS = {
    "Python", "Java", "C++", "Scikit-learn", "TensorFlow", "PyTorch", "NLP",
    "Transformers", "BERT", "SpaCy", "NLTK", "Git", "Docker", "Streamlit",
    "REST APIs", "CNN", "RNN", "GCP", "AWS", "Azure", "Flask", "HuggingFace"
}

# Categorize skill importance
CRITICAL_SKILLS = {"Python", "BERT", "Transformers", "Scikit-learn", "TensorFlow", "PyTorch", "NLP"}
NICE_TO_HAVE = {"Docker", "Git", "Flask", "Streamlit", "GCP", "AWS", "Azure"}

def extract_skills(text):
    text = text.lower()
    skills_found = set()

    for skill in SKILL_KEYWORDS:
        skill_lower = skill.lower().replace(" ", "")
        # Check for skill as word or compressed (e.g., "scikit-learn", "scikitlearn")
        if skill_lower in text.replace("-", "").replace(" ", ""):
            skills_found.add(skill)
    return skills_found


def compute_skill_match(jd_skills, resume_skills):
    if not jd_skills:
        return 0.0
    total_score = 0
    max_score = 0
    for skill in jd_skills:
        weight = 2 if skill in CRITICAL_SKILLS else 1 if skill in NICE_TO_HAVE else 1
        max_score += weight
        if skill in resume_skills:
            total_score += weight
    return total_score / max_score

def rank_resumes_hybrid(jd_embedding, jd_text, resume_embeddings, resume_texts, resume_names, model):
    jd_skills = extract_skills(jd_text)
    results = []
    for i in range(len(resume_embeddings)):
        name = resume_names[i]
        bert_score = util.cos_sim(jd_embedding, resume_embeddings[i]).item()
        resume_skills = extract_skills(resume_texts[i])
        skill_score = compute_skill_match(jd_skills, resume_skills)
        final_score = (bert_score * 0.6) + (skill_score * 0.4)
        results.append((name, final_score, bert_score, skill_score))
    return sorted(results, key=lambda x: x[1], reverse=True)
