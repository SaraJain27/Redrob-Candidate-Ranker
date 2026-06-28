# Redrob Candidate Ranker

AI-powered candidate ranking system developed for the **Redrob Hackathon**.

## Problem

Given a Job Description and a candidate dataset, rank candidates based on their relevance and generate an explainable ranked CSV for recruiters.

The application supports uploaded **CSV** and **JSON** candidate files (≤100 candidates) and produces reproducible rankings suitable for the hackathon sandbox evaluation.

---

## Pipeline

1. Load candidate dataset (CSV or JSON)
2. Parse and preprocess candidate profiles
3. Generate sentence embeddings using **Sentence Transformers (all-MiniLM-L6-v2)**
4. Compute semantic similarity between the Job Description and each candidate
5. Calculate AI Skill Score
6. Calculate Role Relevance Score
7. Combine scores using a hybrid ranking algorithm
8. Generate explainable reasoning for every candidate
9. Export ranked results as a downloadable CSV

---

## Ranking Strategy

Final Score =

* 45% Semantic Similarity
* 25% AI Skill Score
* 15% Role Score
* 15% Production Experience Score

---

## Features

* Upload candidate data in **CSV** or **JSON**
* Supports candidate samples up to **100 candidates**
* Official Job Description support
* Custom Job Description support
* Semantic AI-based ranking
* Explainable candidate reasoning
* Download ranked CSV
* Interactive Gradio interface

---

## Tech Stack

* Python
* Gradio
* Sentence Transformers
* Hugging Face Hub
* Scikit-learn
* Pandas
* NumPy
* python-docx

---

## How to Run

```bash
pip install -r requirements.txt

python app.py
```

---

## Sandbox / Demo

**Hugging Face Space**

(Add your Hugging Face Space link here)

Example:

https://huggingface.co/spaces/sarajain123/Redrob-Candidate-Ranker

---

## Repository Structure

```text
Redrob-Candidate-Ranker/
│── app.py
│── requirements.txt
│── README.md
```



## AI Tools Used

* ChatGPT — architecture design  and code refinement.
* Hugging Face Sentence Transformers — semantic candidate matching.

---

## Author

**Sara Jain**

MCA Student
Thapar Institute of Engineering and Technology
