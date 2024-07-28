import gradio as gr

def greet(name):
    return "Hello " + name + "!"

if __name__ == "__main__":
    with gr.Blocks() as demo:
        
        # 设置输入
        name = gr.Textbox(label="Name")
        
        # 设置输出
        output = gr.Textbox(label="Output Box")
        
        # 设置按钮
        greet_btn = gr.Button("Greet")
        
        # 设置按钮点击事件
        greet_btn.click(fn=greet, inputs=name, outputs=output)
        
    demo.launch()   