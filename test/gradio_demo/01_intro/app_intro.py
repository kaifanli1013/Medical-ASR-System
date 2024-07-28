import gradio as gr

# one in one out
# def greet(name):
#     return "Hello " + name + "!"

# multi
def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)
    
if __name__ == "__main__":
    # demo = gr.Interface(fn=lambda input: input, inputs="text", outputs="text")
    # demo = gr.Interface(
    #     fn=greet,
    #     # 自定义输入框
    #     # 具体设置方法查看官方文档
    #     inputs=gr.Textbox("text", lines=2, placeholder="Name Here...",label="my input"),
    #     outputs="text",
    # )
    # demo.launch()

    # iface = gr.Interface(
    #     fn=greet,
    #     inputs=gr.inputs.Textbox(lines=2, placeholder="Name Here..."),
    #     outputs="text",
    # )
    # app, local_url, share_url = iface.launch()

    demo = gr.Interface(
        fn=greet,
        #按照处理程序设置输入组件
        inputs=["text", "checkbox", gr.Slider(0, 100)],
        #按照处理程序设置输出组件
        outputs=["text", "number"],
    )
    demo.launch()