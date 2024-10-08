import gradio as gr
import pandas as pd
import re
import openai

# 定义提取电子病历的函数
def extract_emr(dialogue):
    response = openai.chat.completions.create(
        model="gpt-4-0613",  # 确保使用支持function calling的模型
        messages=[
            {"role": "system", "content": "あなたは医療アシスタントです。会話から病歴情報を抽出します。"},
            {"role": "user", "content": dialogue}
        ],
        functions=[
            {
                "name": "generate_emr",
                "description": "会話に基づいて電子病歴を生成します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diagnosis": {"type": "string", "description": "診断結果"},
                        "medical_history": {"type": "string", "description": "病歴"},
                        "symptoms": {"type": "string", "description": "症状"},
                    },
                    "required": ["diagnosis"],
                }
            }
        ],
        function_call={"name": "generate_emr"}  # 確実に指定のfunctionを呼び出す
    )

    # 返された結果を処理する
    if response.choices[0].finish_reason == "function_call":
        emr_data = response.choices[0].message["function_call"]["arguments"]
        print(emr_data)
        return emr_data
    else:
        print("Unable to generate EMR")
        return ""

# 解析SRT文件并提取表格
def parse_srt_to_table(file):
    # 读取上传的字幕文件
    with open(file.name, 'r', encoding='utf-8') as srt_file:
        content = srt_file.read()

    # 使用正则表达式提取带有分隔符 '|' 的字幕行
    pattern = r"(SPEAKER_\d+\|.+)"
    matches = re.findall(pattern, content)
    test_response = extract_emr(content)
    print(f"{test_response=}")
    # 构建表格数据
    data = []
    for match in matches:
        # 以 '|' 分割说话者和内容
        speaker, dialogue = match.split('|', 1)

        # 调用提取EMR的功能
        emr = extract_emr(dialogue.strip())
        data.append([speaker.strip(), dialogue.strip(), emr])  # 将提取的EMR添加到表格中
        
    df = pd.DataFrame(data, columns=["Speaker", "Dialogue", "EMR"])
    return df
    

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 从SRT字幕文件生成表格并提取电子病历 (EMR)")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传字幕文件 (.srt)")
            generate_button = gr.Button("生成表格并提取EMR")
        
        with gr.Column():
            output_table = gr.DataFrame(headers=["Speaker", "Dialogue", "EMR"], label="生成的表格")

    # 设置按钮点击事件
    generate_button.click(fn=parse_srt_to_table, inputs=file_input, outputs=output_table)
    
# 启动应用
if __name__ == "__main__":
    demo.launch()
