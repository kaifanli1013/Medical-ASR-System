import os
import sys
import qrcode
import pandas as pd
import gradio as gr

def generate_qr_code(df):
    qr_data = df.to_csv(index=False)
    qr_img = qrcode.make(qr_data, box_size=2, border=2) 
    
    # 定义输出目录
    qr_output_dir = "outputs/qrcode"
    qr_output_dir = os.path.abspath(qr_output_dir)
    print(f"Saving output to {qr_output_dir}")
    
    # 检查并创建输出目录
    if not os.path.exists(qr_output_dir):
        os.makedirs(qr_output_dir)

    # 保存二维码图像
    qr_img_path = os.path.join(qr_output_dir, "qrcode.png")
    qr_img.save(qr_img_path)
    
    csv_output_dir = "outputs/csv"
    csv_output_dir = os.path.abspath(csv_output_dir)
    print(f"Saving output to {csv_output_dir}")
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
        
    # 保存CSV文件
    csv_path = os.path.join(csv_output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    
    # return gr.update(value=csv_path, visible=True), gr.update(value=qr_img_path, visible=True)
    return csv_path, qr_img_path
    
    