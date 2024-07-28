import numpy as py
import gradio as gr

def sepia(input_img):
    sepia_filter = [
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ]
    sepia_img = input_img.dot(sepia_filter)
    sepia_img /= sepia_img.max()  # Normalize values to be between 0 and 1
    return sepia_img

if __name__ == "__main__":
    iface = gr.Interface(fn=sepia, inputs=gr.Image(), outputs="image")
    iface.launch()