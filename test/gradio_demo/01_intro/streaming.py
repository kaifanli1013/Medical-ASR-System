import gradio as gr
import numpy as np

def flip(im):
    return np.flipud(im)

if __name__ == "__main__":
    demo = gr.Interface(
        flip,
        gr.Image(source="webcam", streaming=True),
        "image",
        live=True
    )
    demo.launch()