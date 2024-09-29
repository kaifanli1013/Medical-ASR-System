import pandas as pd
import re
import os
import json
from modules.openai_agent.openai_agent import *

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

def parse_and_summarize(file_list):
    
    # FIXME: 支持多文件
    file = file_list[0]
    
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
