import gradio as gr

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    temperature = f", it is {temperature} degrees today" if temperature else ""
    return f"{salutation}, {name}{temperature}!"
    
demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)

demo.launch()