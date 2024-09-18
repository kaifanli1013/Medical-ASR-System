import gradio as gr

with gr.Blocks() as demo:
    box1 = gr.Textbox(value="Good Job")
    box2 = gr.Textbox(value="Failure")

demo.launch(share=True)