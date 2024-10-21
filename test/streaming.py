# import gradio as gr
# from datetime import datetime

# # 定义一个函数，返回当前的日期和时间。
# def current_time():
#     time_list = []
#     def inner():
#         now = datetime.now()
#         current_time = now.strftime("%Y-%m-%d %H:%M:%S")
#         time_list.append(current_time)
#         return f"欢迎使用,当前时间是: {time_list}\n"
#     return inner
 
# # 使用gr.Blocks创建一个Gradio
# with gr.Blocks() as demo:
              
#     gr.Markdown("# Gradio实时输出的实现")
#     out_1 = gr.Textbox(label="实时状态",
#             value=current_time(),
#             every=1,
#             info="当前时间",)
# # 启动
# demo.launch()

# import gradio as gr
# from transformers import pipeline
# import numpy as np

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cuda")

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
    
#     # Convert to mono if stereo
#     if y.ndim > 1:
#         y = y.mean(axis=1)
        
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]  

# demo = gr.Interface(
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,
# )

# demo.launch()

import gradio as gr
import numpy as np

# 测试函数，用于接收并显示音频片段
def test_audio(new_chunk):
    sr, y = new_chunk
    if y is None:
        return "No audio chunk received"
    
    # 打印采样率和音频片段的形状，方便调试
    return f"Sample rate: {sr}, Audio chunk shape: {y.shape}"

# 测试 Gradio 的 Audio streaming 功能
demo = gr.Interface(
    test_audio,  # 测试函数
    [gr.Audio(sources=["microphone"], streaming=True)],  # 输入为流式音频
    "text",  # 输出为文本
    live=True
)

demo.launch()
