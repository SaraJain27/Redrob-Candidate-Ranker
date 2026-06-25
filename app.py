import gradio as gr
import pandas as pd

# Load ranked candidates
df = pd.read_csv("final_df.csv")

# Sort by final score
df = df.sort_values("final_score_v3", ascending=False)

def rank_candidates(job_description):

    top = df.head(100)[[
        "candidate_id",
        "title",
        "final_score_v3",
        "reasoning"
    ]].copy()

    top.rename(columns={
        "final_score_v3": "score"
    }, inplace=True)

    return top

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.Textbox(
        lines=10,
        label="Paste Job Description"
    ),
    outputs=gr.Dataframe(
        label="Top 100 Ranked Candidates"
    ),
    title="AI Recruiter System",
    description="""
Paste a job description to view the top-ranked candidates.
The ranking is based on semantic similarity, production experience,
behavior signals, and professional experience.
"""
)

demo.launch()
