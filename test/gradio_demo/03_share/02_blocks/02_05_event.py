# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!"

# with gr.Blocks() as demo:
#     name = gr.Textbox(label="Name")
#     output = gr.Textbox(label="Output", interactive=True)
#     greet_btn = gr.Button("Greet")
#     greet_btn.click(fn=greet, inputs=name, outputs=output)
    
# demo.launch()

# import gradio as gr
# def welcome(name):
#     return f"Welcome to Gradio, {name}!"
# with gr.Blocks() as demo:
#     gr.Markdown(
#     """
#     # Hello World!
#     Start typing below to see the output.
#     """)
#     inp = gr.Textbox(placeholder="What is your name?")
#     out = gr.Textbox()
#     #设置change事件
#     inp.change(fn = welcome, inputs = inp, outputs = out)
# demo.launch()

# import gradio as gr
# def increase(num):
#     return num + 1
# with gr.Blocks() as demo:
#     a = gr.Number(label="a")
#     b = gr.Number(label="b")
#     # 要想b>a，则使得b = a+1
#     atob = gr.Button("b > a")
#     atob.click(increase, a, b)
#     # 要想a>b，则使得a = b+1
#     btoa = gr.Button("a > b")
#     btoa.click(increase, b, a)
# demo.launch()

# import gradio as gr

# with gr.Blocks() as demo:
#     food_box = gr.Number(value=10, label="Food Count")
#     status_box = gr.Textbox(label="Status")
    
#     def eat(food):
#         if food > 0:
#             return food - 1, "full"
#         else:
#             return 0, "hungry"
#     gr.Button("EAT").click(
#         fn=eat,
#         inputs=food_box,
#         outputs=[food_box, status_box]
#     )
    
# demo.launch()

import gradio as gr
def change_textbox(choice):
    
    if choice == "short":
        return gr.update(lines=2, visible=True, value="Short story: ")
    elif choice == "long":
        return gr.update(lines=8, visible=True, value="Long story: ")
    else:
        return gr.update(visible=False)
    
with gr.Blocks() as demo:
    radio = gr.Radio(["short", "long", "none"], label="Essay Length to Write")
    text = gr.Textbox(lines=2, interactive=True)
    radio.change(change_textbox, inputs=radio, outputs=text)
    
demo.launch()