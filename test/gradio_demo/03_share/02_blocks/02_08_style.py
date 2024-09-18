# import gradio as gr
# #修改blocks的背景颜色
# with gr.Blocks(css=".gradio-container {background-color: red}") as demo:
#     box1 = gr.Textbox(value="Good Job")
#     box2 = gr.Textbox(value="Failure")
# demo.launch()

import gradio as gr
# 这里用的是id属性设置
with gr.Blocks(css="#warning {background-color: red}") as demo:
    box1 = gr.Textbox(value="Good Job", elem_id="warning")
    box2 = gr.Textbox(value="Failure")
    box3 = gr.Textbox(value="None", elem_id="warning")
demo.launch()