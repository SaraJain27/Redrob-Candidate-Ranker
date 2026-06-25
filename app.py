import gradio as gr
import pandas as pd

# Load ranked candidates
df = pd.read_csv("final_df.csv")

# Sort by your best score
df = df.sort_values("final_score_v3", ascending=False)

def rank_candidates(job_description):

    top = df.head(20)[[
        "candidate_id",
        "title",
        "semantic_score",
        "behavior_score",
        "production_evidence_norm",
        "professional_score_norm",
        "final_score_v3"
    ]].copy()

    top = top.rename(columns={
        "final_score_v3": "score"
    })

    return top

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.Textbox(
        lines=10,
        label="Paste Job Description"
    ),
    outputs=gr.Dataframe(),
    title="AI Recruiter System",
    description="Rank candidates using semantic matching, behavior signals, production evidence and professional experience."
)

demo.launch()