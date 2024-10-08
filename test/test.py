import gradio as gr
import pyaudio
import websockets
import asyncio

# 实时录音功能
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    print("Recording...")
    audio_frames = []
    for _ in range(0, int(16000 / 1024 * 5)):  # 5 秒录音
        data = stream.read(1024)
        audio_frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return audio_frames

# Gradio 界面
def process_audio():
    audio_data = record_audio()
    # 这里可以将音频数据发送给 WebSocket 或者进行其他处理
    return "Audio recorded and processed."

iface = gr.Interface(fn=process_audio, inputs=None, outputs="text", live=True)
iface.launch()
