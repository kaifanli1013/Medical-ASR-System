import random
import gradio as gr

def chat(message, history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = random.randint(1, 10)
    elif message.startswith("how"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history

if __name__ == "__main__":
    chatbot = gr.Chatbot() # .style(color_map=("green", "pink"))
    chatbot.style(message_color="green", response_color="pink")
    demo = gr.Interface(
        chat,
        ["text", "state"],
        [chatbot, "state"],
        # 设置没有保存数据的按钮
        allow_flagging="never",
    )
    demo.launch()
        