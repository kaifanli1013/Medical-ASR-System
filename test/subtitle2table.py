import gradio as gr
import pandas as pd
import re
import os
import json
import sys
import openai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.openai_agent.openai_agent import *

# 替换为你自己的 OpenAI API key
openai.api_key = "your-openai-api-key"

field_mapping = {
    'presenting_complaint': '主訴',
    'symptoms': '症状',
    'physical_examination_findings': '身体所見',
    'test_results': '検査結果',
    'diagnosis': '診断',
    'treatment_plan': '治療計画',
    'prescription_info': '処方情報',
    'follow_up': '経過観察'
}

def parse_srt_to_table(content):
    pattern = r"(SPEAKER_\d+\|.+)"
    matches = re.findall(pattern, content)

    data = []
    for match in matches:
        speaker, dialogue = match.split('|', 1)
        data.append([speaker.strip(), dialogue.strip(), ""])  # EMR列为空
        
    df = pd.DataFrame(data, columns=["Speaker", "Dialogue", "EMR"])
    return df

def summarize_conversation(content):
    system_message = "You are a medical assistant. You are talking to a patient who is describing their symptoms to you. You need to summarize the conversation into the patient's electronic medical record."
    conversation=Conversation()
    conversation.add_message("system", system_message)
    conversation.add_message("user", content)  # 将整个对话传入

    # 调用OpenAI的API获取总结
    chat_response = chat_completion_request(conversation.conversation_history, tools=tools)
    
    summarized_emr = chat_response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(summarized_emr)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response.data[0].embedding  # 修正
    
    # 计算每一行对话的相关性
    strings_and_relatednesses = [
        (row["Dialogue"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    
    # 打印调试信息
    # print("strings_and_relatednesses:", strings_and_relatednesses)
    
    if not strings_and_relatednesses:  # 如果为空，抛出异常或处理
        raise ValueError("No relatedness scores were computed. Check your embeddings or data.")
    
    # 按相似度排序
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)  # 解包结果
    return strings[:top_n]

def calculate_embeddings_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """为每个对话行计算嵌入并添加到DataFrame中."""
    df['embedding'] = df['Dialogue'].apply(lambda x: embedding_request(x).data[0].embedding)
    # print("Generated embeddings:", df['embedding'])
    return df

def parse_and_summarize(file):
    
    with open(file.name, 'r', encoding='utf-8') as srt_file:
        content = srt_file.read()
    
    # 假设文件处理完毕，并生成了表格
    output_table = parse_srt_to_table(content)  # 假设你已经有parse_srt_to_table函数
    
    output_table = calculate_embeddings_for_df(output_table)  # 计算每行的嵌入
    
    # 获取电子病历的摘要
    summarized_emr = summarize_conversation(content)
    print(f"{summarized_emr=}")
    
    # 遍历EMR的每一部分并与对话匹配
    emr_sections = {
        'presenting_complaint': summarized_emr.get('presenting_complaint', []),
        'symptoms': summarized_emr.get('symptoms', []),
        'physical_examination_findings': summarized_emr.get('physical_examination_findings', []),
        'test_results': summarized_emr.get('test_results', []),
        'diagnosis': [summarized_emr.get('diagnosis', '')],  # 单条诊断
        'treatment_plan': summarized_emr.get('treatment_plan', []),
        'prescription_info': summarized_emr.get('prescription_info', []),
        'follow_up': summarized_emr.get('follow_up', [])
    }

    for section, entries in emr_sections.items():
        # 使用映射表将section转换为日文
        japanese_section = field_mapping.get(section, section)  # 默认返回原始section名称，如果没有映射
        for entry in entries:
            most_related_dialogues = strings_ranked_by_relatedness(entry, output_table, top_n=1)
            for dialogue in most_related_dialogues:
                idx = output_table[output_table['Dialogue'] == dialogue].index[0]
                
                # 获取当前EMR列的已有内容
                current_emr = output_table.at[idx, 'EMR']
                
                # 如果当前EMR列已有内容，则追加新的内容，否则直接设置为新内容
                if current_emr:
                    output_table.at[idx, 'EMR'] = f"{current_emr}\n{japanese_section}: {entry}"
                else:
                    output_table.at[idx, 'EMR'] = f"{japanese_section}: {entry}"
                    
    return output_table.drop(columns=['embedding'])

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 从SRT字幕文件生成表格")
    
    with gr.Column():
        file_input = gr.File(label="上传字幕文件 (.srt)")
        generate_button = gr.Button("生成表格")
        
    with gr.Column():
        output_table = gr.DataFrame(headers=["Speaker", "Dialogue", "EMR"], label="生成的表格", interactive=True)
        
    generate_button.click(fn=parse_and_summarize, inputs=file_input, outputs=output_table)

if __name__ == "__main__":
    demo.launch()
