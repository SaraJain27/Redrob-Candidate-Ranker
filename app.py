import tempfile
import json
import os
import re

import gradio as gr
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download
from docx import Document

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading official Job Description...")

jd_path = hf_hub_download(
    repo_id="sarajain123/Redrob-Dataset",
    repo_type="dataset",
    filename="job_description.docx"
)

doc = Document(jd_path)

OFFICIAL_JD = "\n".join(
    p.text
    for p in doc.paragraphs
)

TITLE_COL = "title"
TEXT_COL = "candidate_text"
BEHAVIOR_COL = "behavior_score"
PRODUCTION_COL = "production_evidence_norm"
PROFESSIONAL_COL = "professional_score_norm"


def load_candidates(candidate_file):

    ext = os.path.splitext(candidate_file.name)[1].lower()

    if ext == ".csv":

        df = pd.read_csv(candidate_file.name)

        for col in [
            "behavior_score",
            "production_evidence_norm",
            "professional_score_norm"
        ]:

            if col not in df.columns:
                df[col] = 0.5

        return df

    with open(candidate_file.name, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for cand in data:

        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})

        summary = profile.get("summary", "")

        history = " ".join(
            job.get("description", "")
            for job in career
        )

        skill_text = " ".join(
            skill.get("name", "")
            for skill in skills
        )

        rows.append({

            "candidate_id":
                cand.get("candidate_id", ""),

            "title":
                profile.get("current_title", ""),

            "candidate_text":
                summary + " " + history + " " + skill_text,

            "behavior_score":
                signals.get(
                    "interview_completion_rate",
                    0.5
                ),

            "production_evidence_norm":
                min(
                    profile.get(
                        "years_of_experience",
                        0
                    ) / 10,
                    1
                ),

            "professional_score_norm":
                signals.get(
                    "profile_completeness_score",
                    50
                ) / 100

        })

    return pd.DataFrame(rows)
# ==========================================================
# ROLE SCORE
# ==========================================================

def role_score(title, text):

    text = (str(title) + " " + str(text)).lower()

    score = 0

    # Strong AI keywords
    positive = {

        "ai engineer":5,
        "machine learning":5,
        "ml engineer":5,
        "deep learning":5,
        "nlp":4,
        "llm":5,
        "rag":5,
        "langchain":4,
        "semantic search":4,
        "retrieval":4,
        "vector database":4,
        "faiss":4,
        "pinecone":4,
        "qdrant":4,
        "weaviate":4,
        "tensorflow":4,
        "pytorch":4,
        "python":3,
        "transformers":4,
        "recommendation":3,
        "search engineer":3,
        "applied scientist":4,
        "data scientist":3
    }

    for word, value in positive.items():

        if word in text:
            score += value

    # Strong negative keywords

    negative = {

        "mechanical":-8,
        "civil":-8,
        "graphic designer":-8,
        "designer":-6,
        "project manager":-6,
        "sales":-8,
        "marketing":-8,
        "accountant":-8,
        "teacher":-8,
        "customer support":-8,
        "business analyst":-4

    }

    for word, value in negative.items():

        if word in text:
            score += value

    return score

# ==========================================================
# REASONING
# ==========================================================

# ==========================================================
# AI SKILL SCORE
# ==========================================================

def skill_score(text):

    text = str(text).lower()

    score = 0

    skills = {

        "python":2,
        "tensorflow":3,
        "pytorch":3,
        "keras":2,
        "scikit-learn":2,
        "machine learning":4,
        "deep learning":4,
        "nlp":4,
        "llm":5,
        "rag":5,
        "langchain":4,
        "transformers":4,
        "huggingface":3,
        "embedding":3,
        "embeddings":3,
        "semantic search":4,
        "retrieval":4,
        "faiss":4,
        "pinecone":4,
        "qdrant":4,
        "weaviate":4,
        "vector database":4,
        "recommendation":2,
        "search":2
    }

    for skill, value in skills.items():

        if skill in text:

            score += value

    return score


def create_reason(row):

    title = str(row[TITLE_COL])

    text = str(row[TEXT_COL]).lower()

    reason = ""

    years = re.search(
        r"(\d+(\.\d+)?)\+?\s*years",
        text
    )

    if years:

        reason += (
            f"{title} with "
            f"{years.group(1)}+ years of experience. "
        )

    else:

        reason += f"{title}. "

    if "nlp" in text:

        reason += "Strong NLP experience. "

    elif "machine learning" in text:

        reason += "Strong Machine Learning experience. "

    elif "deep learning" in text:

        reason += "Strong Deep Learning experience. "

    elif "backend" in text:

        reason += "Strong backend engineering experience. "

    elif "computer vision" in text:

        reason += "Computer Vision experience. "

    technologies = []

    tech_list = [

        "python",
        "tensorflow",
        "pytorch",
        "langchain",
        "llm",
        "rag",
        "retrieval",
        "ranking",
        "semantic search",
        "embeddings",
        "faiss",
        "pinecone",
        "qdrant",
        "weaviate",
        "opensearch"

    ]

    for tech in tech_list:

        if tech in text:

            technologies.append(
                tech.title()
            )

    technologies = list(dict.fromkeys(technologies))

    if technologies:

        reason += (

            "Skills: "

            + ", ".join(
                technologies[:5]
            )

            + ". "

        )

    if row[PRODUCTION_COL] > 0.70:

        reason += "Strong production AI experience. "

    if row[PROFESSIONAL_COL] > 0.70:

        reason += "Professional profile is strong. "

    reason += (
        "Overall, candidate is well aligned with the role."
    )

    return reason
# ==========================================================
# RANK CANDIDATES
# ==========================================================

def rank_candidates(candidate_file, job_description):

    if candidate_file is None:
        raise gr.Error("Please upload a CSV or JSON file.")

    # -----------------------------
    # Load uploaded candidates
    # -----------------------------

    results = load_candidates(candidate_file)

    # Required columns
    required = [
        "candidate_id",
        "candidate_text",
        "title"
    ]

    for col in required:
        if col not in results.columns:
            raise gr.Error(f"Missing required column: {col}")

    # Default values if missing
    defaults = {
        "behavior_score": 0.5,
        "production_evidence_norm": 0.5,
        "professional_score_norm": 0.5
    }

    for col, value in defaults.items():
        if col not in results.columns:
            results[col] = value

    # -----------------------------
    # Official JD
    # -----------------------------

    if job_description.strip() == "":
        job_description = OFFICIAL_JD

    # -----------------------------
    # Generate embeddings
    # -----------------------------

    candidate_embeddings = model.encode(
        results["candidate_text"].fillna("").tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    jd_embedding = model.encode(
        job_description,
        normalize_embeddings=True
    )

    # -----------------------------
    # Semantic Similarity
    # -----------------------------

    semantic_scores = util.cos_sim(
        jd_embedding,
        candidate_embeddings
    )[0].cpu().numpy()

    scaler = MinMaxScaler()

    results["semantic_score"] = scaler.fit_transform(
        semantic_scores.reshape(-1, 1)
    )
   
 
    # -----------------------------
    # Role Score
    # -----------------------------

    results["role_score"] = results.apply(
        lambda x: role_score(
            x["title"],
            x["candidate_text"]
        ),
        axis=1
    )

    results["role_score"] = scaler.fit_transform(
        results[["role_score"]]
    )

    # -----------------------------
    # Skill Score
    # -----------------------------

    results["skill_score"] = results["candidate_text"].apply(
        skill_score
    )

    results["skill_score"] = scaler.fit_transform(
        results[["skill_score"]]
    )
    # -----------------------------
    # Final Score
    # -----------------------------

    results["final_score"] = (

       0.45 * results["semantic_score"]

       + 0.25 * results["skill_score"]

       + 0.15 * results["role_score"]

       + 0.15 * results["production_evidence_norm"]

    )

    # -----------------------------
    # Sort
    # -----------------------------

    results = results.sort_values(
        "final_score",
        ascending=False
    ).copy()

    # -----------------------------
    # Rank
    # -----------------------------

    results["rank"] = range(
        1,
        len(results) + 1
    )

    results["score"] = results["final_score"].round(6)

    # -----------------------------
    # Reasoning
    # -----------------------------

    results["reasoning"] = results.apply(
        create_reason,
        axis=1
    )

    submission = results[
        [
            "candidate_id",
            "rank",
            "score",
            "reasoning"
        ]
    ]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    submission.to_csv(
        temp_file.name,
        index=False
    )

    return submission, temp_file.name
# ==========================================================
# GRADIO INTERFACE
# ==========================================================

demo = gr.Interface(
    fn=rank_candidates,

    inputs=[
        gr.File(
            label="Upload Candidate CSV or JSON (≤100 Candidates)"
        ),
        gr.Textbox(
            lines=12,
            label="Job Description",
            placeholder="Leave empty to use the official Job Description"
        )
    ],

    outputs=[
        gr.Dataframe(
            label="Ranked Candidates"
        ),
        gr.File(
            label="Download Ranked CSV"
        )
    ],

    title="AI Candidate Ranker",

    description="""
Upload a candidate CSV or Redrob JSON (up to 100 candidates),
enter a Job Description (or leave it blank to use the official one),
and download the ranked candidate CSV.
"""
)

if __name__ == "__main__":
    demo.launch()
