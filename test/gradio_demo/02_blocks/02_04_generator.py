import gradio as gr
import numpy as np
import time

def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.randint(255, size=(300, 600, 3))
        yield image
        
        
if __name__ == "__main__":
    demo = gr.Interface(
        fake_diffusion,
        gr.Slider(1, 10, value=3, step=1),
        "image",
        live=True
    )
    #生成器必须要queue函数
    demo.queue()
    demo.launch()