import torchkeras 
from torchkeras.data import download_baidu_pictures 
download_baidu_pictures('猫咪表情包',100)

import gradio as gr
from PIL import Image
import time,os
from pathlib import Path 
base_dir = '猫咪表情包'
selected_dir = 'selected'
files = [str(x) for x in 
         Path(base_dir).rglob('*.jp*g') 
         if 'checkpoint' not in str(x)]
def show_img(path):
    return Image.open(path)
def fn_before(done,todo):
    ...
    return done,todo,path,img
def fn_next(done,todo):
    ...
    return done,todo,path,img
def save_selected(img_path):
    ...
    return msg 
def get_default_msg():
    ...
    return msg
    
    
with gr.Blocks() as demo:
    with gr.Row():
        total = gr.Number(len(files),label='总数量')
        with gr.Row(scale = 1):
            bn_before = gr.Button("上一张")
            bn_next = gr.Button("下一张")
        with gr.Row(scale = 2):
            done = gr.Number(0,label='已完成')
            todo = gr.Number(len(files),label='待完成')
    path = gr.Text(files[0],lines=1, label='当前图片路径')
    feedback_button = gr.Button("选择图片",variant="primary")
    msg = gr.TextArea(value=get_default_msg,lines=3,max_lines = 5)
    img = gr.Image(value = show_img(files[0]),type='pil')
    
    bn_before.click(fn_before,
                 inputs= [done,todo], 
                 outputs=[done,todo,path,img])
    bn_next.click(fn_next,
                 inputs= [done,todo], 
                 outputs=[done,todo,path,img])
    feedback_button.click(save_selected,
                         inputs = path,
                         outputs = msg
                         )

demo.launch()