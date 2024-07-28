import gradio as gr

# 输入文本处理程序
def greet(name):
    return "Hello " + name + "!"


if __name__ == "__main__":
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch()