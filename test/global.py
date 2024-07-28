import gradio as gr

scores = []

def track_score(score):
    scores.append(score)
    # return top 3 scores
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores

if __name__ == "__main__":
    demo = gr.Interface(
        track_score,
        gr.Number(label="Score"),
        gr.JSON(label="Top 3 Scores"),
    )
    demo.launch()