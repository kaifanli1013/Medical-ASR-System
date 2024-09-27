import gradio as gr
import pandas as pd
import re

def parse_srt_to_table(file):
    # 读取上传的字幕文件
    with open(file.name, 'r', encoding='utf-8') as srt_file:
        content = srt_file.read()

    # 使用正则表达式提取带有分隔符 '|' 的字幕行
    pattern = r"(SPEAKER_\d+\|.+)"
    matches = re.findall(pattern, content)

    # 构建表格数据
    data = []
    for match in matches:
        # 以 '|' 分割说话者和内容
        speaker, dialogue = match.split('|', 1)
        data.append([speaker.strip(), dialogue.strip(), ""])  # EMR列为空
        
    df = pd.DataFrame(data, columns=["Speaker", "Dialogue", "EMR"])
    # print(df)
    return df
    

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 从SRT字幕文件生成表格")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传字幕文件 (.srt)")
            generate_button = gr.Button("生成表格")
        
        with gr.Column():
            output_table = gr.DataFrame(headers=["speaker", "content", "EMR"], label="生成的表格")

    # 设置按钮点击事件
    generate_button.click(fn=parse_srt_to_table, inputs=file_input, outputs=output_table)

# 启动应用
if __name__ == "__main__":
    demo.launch()