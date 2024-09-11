import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=1, placeholder="Name here...", label="my input"),
    outputs="text",
)


# demo.launch()

if __name__ == "__main__":
    app, local_url, share_url = demo.launch()