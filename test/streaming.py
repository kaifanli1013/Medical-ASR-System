import gradio as gr
from datetime import datetime

# 定义一个函数，返回当前的日期和时间。
def current_time():
    time_list = []
    def inner():
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        time_list.append(current_time)
        return f"欢迎使用,当前时间是: {time_list}\n"
    return inner
 
# 使用gr.Blocks创建一个Gradio
with gr.Blocks() as demo:
              
    gr.Markdown("# Gradio实时输出的实现")
    out_1 = gr.Textbox(label="实时状态",
            value=current_time(),
            every=1,
            info="当前时间",)
# 启动
demo.launch()