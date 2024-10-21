import gradio as gr
import time

def long_running_task():
    for i in range(10):
        time.sleep(1)  # 模拟计算时间
        yield f"Processing step {i+1}/10 completed.\n"

# 创建Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        output_box = gr.Textbox(label="Real-time Output", lines=20)
    
    # 启动长时间任务的按钮
    run_button = gr.Button("Start Task")
    
    # 绑定回调函数，将结果逐步显示在output_box
    run_button.click(fn=long_running_task, outputs=output_box, streaming=True)

demo.launch()
