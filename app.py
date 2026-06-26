import gradio as gr
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = "all-MiniLM-L6-v2"
REPO_ID = "sarajain123/Redrob-Dataset"

model = SentenceTransformer(MODEL_NAME)

candidate_path = hf_hub_download(REPO_ID, repo_type="dataset", filename="candidate_text_df.csv")
embedding_path = hf_hub_download(REPO_ID, repo_type="dataset", filename="candidate_embeddings.npy")
final_path = hf_hub_download(REPO_ID, repo_type="dataset", filename="final_df.csv")

candidate_df = pd.read_csv(candidate_path)
final_df = pd.read_csv(final_path)
candidate_embeddings = np.load(embedding_path)

df = candidate_df.merge(final_df, on="candidate_id", how="left")

def role_score(text):
    text = str(text).lower()
    score = 0
    positive = [
        "ai engineer","machine learning","ml engineer","deep learning",
        "retrieval","ranking","recommendation","search","nlp",
        "embedding","vector","faiss","pinecone","weaviate",
        "qdrant","milvus","llm","python"
    ]
    negative = [
        "civil engineer","mechanical engineer","electrical engineer",
        "marketing","sales","hr","accountant","teacher","doctor"
    ]
    for p in positive:
        if p in text:
            score += 1
    for n in negative:
        if n in text:
            score -= 2
    return score

def create_reason(row):

    title = str(row["title"])
    text = str(row["candidate_text"]).lower()

    strengths = []

    keyword_map = {
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "python": "Python",
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "llm": "Large Language Models",
        "rag": "Retrieval-Augmented Generation (RAG)",
        "retrieval": "Retrieval Systems",
        "ranking": "Ranking Pipelines",
        "recommendation": "Recommendation Systems",
        "embedding": "Embedding Models",
        "vector": "Vector Search",
        "faiss": "FAISS",
        "pinecone": "Pinecone",
        "langchain": "LangChain",
        "nlp": "Natural Language Processing"
    }

    for key, value in keyword_map.items():
        if key in text:
            strengths.append(value)

    strengths = list(dict.fromkeys(strengths))

    reason = f"The candidate is currently working as {title}. "

    if strengths:
        reason += (
            "Their profile demonstrates hands-on experience in "
            + ", ".join(strengths[:5])
            + ", indicating strong technical alignment with the job requirements. "
        )

    if row["production_evidence_norm_y"] > 0.70:
        reason += (
            "The profile also reflects meaningful production-level experience, "
            "suggesting the ability to build and deploy scalable AI solutions. "
        )

    if row["behavior_score"] > 0.70:
        reason += (
            "In addition, the candidate has strong recruiter engagement and "
            "positive behavioural indicators, improving hiring confidence. "
        )

    if row["professional_score_norm_y"] > 0.70:
        reason += (
            "Overall, the professional background and technical expertise make "
            "this candidate a strong match for the role."
        )
    else:
        reason += (
            "Overall, the candidate shows good potential and aligns well with "
            "the core technical expectations of the position."
        )

    return reason

def rank_candidates(job_description):
    jd_embedding = model.encode([job_description], normalize_embeddings=True)
    similarity = cosine_similarity(jd_embedding, candidate_embeddings)[0]

    results = df.copy()
    results["semantic_score"] = similarity
    results["role_score"] = results.apply(
        lambda r: role_score(str(r["title"]) + " " + str(r["candidate_text"])),
        axis=1,
    )

    scaler = MinMaxScaler()
    results["semantic_score"] = scaler.fit_transform(results[["semantic_score"]])
    results["role_score"] = scaler.fit_transform(results[["role_score"]])

    results["final_score_v4"] = (
        0.35 * results["semantic_score"]
        + 0.25 * results["role_score"]
        + 0.15 * results["production_evidence_norm_y"]
        + 0.10 * results["behavior_score"]
        + 0.15 * results["professional_score_norm_y"]
    )

    results = results.sort_values("final_score_v4", ascending=False).head(100).copy()
    results["rank"] = range(1, len(results) + 1)
    results["reasoning"] = results.apply(create_reason, axis=1)

    return results[
        ["rank","candidate_id","title","final_score_v4","reasoning"]
    ].rename(columns={"final_score_v4":"score"})

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.Textbox(lines=12, label="Job Description"),
    outputs=gr.Dataframe(),
    title="AI Candidate Ranker",
    description="Ranks candidates against the supplied job description."
)

if __name__ == "__main__":
    demo.launch()
