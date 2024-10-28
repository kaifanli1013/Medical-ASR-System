import os
import argparse
import json
import torch
import pandas as pd
import textstat
import logging
from transformers import pipeline

if not os.path.exists('./output'):
    os.makedirs('./output')
logging.basicConfig(filename='./output/results.log', level=logging.INFO, format='%(message)s', filemode='w')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

five_shot_examples = """
    以下は、いくつの例です。これらの例は、入力文と出力の例を示しています。これらの例は、入力文から医療情報を抽出する方法を示しています。
    例1:
        入力: 最近、すごく喉が渇いて、水をたくさん飲んでしまいます。そのためにトイレに行く回数も増えて、疲れやすくなっています。
        出力: 現病歴:\\n-頻繁な喉の渇きと頻尿
    例2:
        入力: はい、体重が急に減りましたし、時々手足が震える感じもあります。
        出力: 現病歴:\\n-体重減少と疲労感\\n-時々の手足の震え
    例3:
        入力: 最近、胸にしこりを感じるようになって、痛みも少しあります。心配になって病院に来ました。
        出力: 現病歴:\\n-乳房にしこりを感じる
    例4:
        入力: こんにちは。今日はどのようなご相談ですか？
        出力: -
        補足説明: 医療情報が含まれていないため、出力は「-」としてください。
    例5:
        入力: はい、尿の量が減って、色も濃くなってきました。夜間にもトイレに行く回数が増えました。
        出力: 現病歴:\\n-尿量減少、尿色の濃化、頻尿、夜間尿
"""

system_prompt_json_format = """
あなたは医療アシスタントです。医師と患者、または患者の家族との対話から、以下の重要な医療情報を抽出し、電子カルテに記録してください
    - 現病歴
    - 既往歴
    - 診断
    - 症状
    - 治療計画
    - 家族歴
    - 生活歴
        
出力のフォーマットは以下のようなkey-valueペア形式で記述してください。
各keyは情報のカテゴリー(例: 診断、症状、家族歴、既往歴)を示し、各value具体的な情報のリストとして記述します。

    フォーマットは以下のようです:
        {
            "現病歴": ...,
            "既往歴": ...,
            "診断": ...,
            "症状": ...,
            "治療計画": ...,
            "家族歴": ...,
            "生活歴": ...,
        }

    
ただし、以下の制約に注意してください:
1. 各keyは以下のkey_listに限定されており、**必ずリスト内のものを使用してください**。リストにないkeyは使用しないでください。
    
    key_list = [
        '現病歴',
        '既往歴',
        '診断',
        '症状',
        '治療計画',
        '家族歴',
        '生活歴',
    ]
    
2. 入力文から医療情報が抽出できない場合は勝手な情報を出力しないでください。出力は入力文に基づいて正確な情報を記述してください。

3. 出力のフォーマットに従って記述してください。出力には、上記のフォーマット以外のテキストを含めないでください。

"""

def post_process(output: dict):
    final_output = ""
    for key, value in output.items():
        if value:
            final_output += f"{key}:\n"
            for v in value:
                final_output += f"- {v}\n"
    if not final_output:
        final_output = "-"
    return final_output

def show_metrics(result_list: list):
    fk_score_avg = 0
    smog_score_avg = 0
    dale_chall_score_avg = 0
    
    for result in result_list:
        fk_score = textstat.flesch_kincaid_grade(result)
        smog_score = textstat.smog_index(result)
        dale_chall_score = textstat.dale_chall_readability_score(result)
        
        fk_score_avg += fk_score
        smog_score_avg += smog_score
        dale_chall_score_avg += dale_chall_score
    
    fk_score_avg /= len(result_list)
    smog_score_avg /= len(result_list)
    dale_chall_score_avg /= len(result_list)
    
    logging.info(f"FK Score: {fk_score_avg}")
    logging.info(f"SMOG Score: {smog_score_avg}")
    logging.info(f"Dale Chall Score: {dale_chall_score_avg}")
    
    return (fk_score_avg, smog_score_avg, dale_chall_score_avg)
        
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("model_type", type=str, choices=["meta", "rinna"], nargs='?', default="meta", help="Choose model type, default is 'meta'")
    args.add_argument("model_size", type=str, choices=["1B", "3B"], nargs='?', default="3B", help="Choose model size, default is '1B'")
    args.add_argument("few_shot", type=str, choices=['0-shot', '1-shot', '5-shot', '10-shot', '20-shot'], nargs='?', default='0-shot', help="Enable few-shot learning, default is False")
    args.add_argument("use_memory", type=bool, nargs='?', const=True, default=False, help="Use memory, default is False")
    args = args.parse_args()
    
    # load model
    if args.model_type == "meta":
        if args.model_size == "1B":
            model_id = "meta-llama/Llama-3.2-1B-Instruct"
        elif args.model_size == "3B":
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
        elif args.model_size == "8B":
            model_id = "meta-llama/Llama-3-8B-Instruct"
        else:
            raise ValueError("Invalid model size")
    elif args.model_type == "rinna":
        model_id = "rinna/llama-3-youko-8b-instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    base_message = "この文から医療情報を抽出してください: "

    # load input
    input_file = "./data/gpt_generated_medical_dialogue.xlsx"
    df = pd.read_excel(input_file)

    ground_truth = df[8:]['report- text']
    generated_text = []
    
    if args.enable_few_shot:
        system_prompt_json_format += five_shot_examples
        
    for i, row in df[8:10].iterrows():
        user_message = base_message + row['speech- text']
        messages = [
            {"role": "system", "content": system_prompt_json_format},
            {"role": "assistant", "content": user_message},
        ]
        
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )

        try:
            output = json.loads(outputs[0]["generated_text"][-1]['content'])
            logging.info(f'Row: {i}')
            logging.info(f"input: {user_message}")
            logging.info(f"output: {post_process(output)}")
            logging.info(f"ground truth: {ground_truth[i]}")
            logging.info("\n")
            generated_text.append(post_process(output))
        except:
            generated_text.append("-")
            logging.DEBUG("Error in processing output")
            logging.info(outputs[0]["generated_text"][-1]['content'])
        
    
    logging.info("Readability scores for generated text:")
    logging.info(show_metrics(generated_text))

        

    
# system_prompt_en = """
# You are a medical report generation assistant. You are given a single line from a patient-doctor conversation, which may 
# contain relevant medical information. Your task is to analyze this line and determine if it contains inforamtion that 
# should be summarized in a medical report entry. If so, you should generate a concise medical report entry based on the
# information contained in the text.

# Instructions:
# 1.  The format restrictions for the output are as follows:
#     1.1 The output should focus on key points such as Diagnosis, Symptoms, Family History, 
#         and any other relevant information like Dietary Habits or Lifestyle if available in the input.
#     1.2 Only include a section if the input contains relevant information for that section. 
#         If there is no information about a particular category (e.g., Diagnosis or Symptoms), 
#         omit that category from the output.
#     1.3 If the input text does not contain relevant medical information, you should return '-' in the report entry.
#     1.4 The example output format is as follows:
#         example 1:
#             Diagnosis: 
#                 - Hypertension
#             Symptoms:
#                 - Headache
#                 - dizziness
#             Family History:
#                 None
#         example 2:
#             Symptoms:
#                 - Fatigue
#                 - Chest pain
#             Dietary Habits:
#                 - High salt intake
#                 - Skips breakfast regularly   
    
# 2.  Funtion calling restrictions are as follows:
#     2.1 You will need to make one or more function/tool calls to achieve the purpose. 
#     2.2 If none of the functions can be used, point it out. 
#     2.3 If the given text lacks the parameters required by the function, also point it out.
#     2.4 You should only return the function call in tools call sections.
#     2.5 If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] 
#         You SHOULD NOT include any other text in the response.

#     Here is a list of functions in JSON format that you can invoke.[
#         {
#             "name": "generate_medical_report",
#             "description": "Analyze a single line of patient-doctor conversation and generate a medical report entry based on the content.",
#             "parameters": {
#                 "type": "dict",
#                 "required": [
#                     "contains_information",
#                     "report_text",
#                 ],
#                 "properties": {
#                     "contains_information": {
#                         "type": "boolen",
#                         "description": "Indicates whether the input line contains relevant medical information and should be 
#                         summarized in the report. If true, generate a report entry in 'report_text'; 
#                         if false, return '-' in 'report_text'."
#                     },
#                     "report_text": {
#                         "type": "string",
#                         "description": "The generated medical report entry based on the input text."
#                     }
#                 },
#             }
#         }
#     ]
    
# # Few-shot examples:
# # {few_shot_examples}
# # """

# system_prompt_jp = f"""
# あなたは医療アシスタントです。医師と患者、または患者の家族との対話から、以下の重要な医療情報を抽出し、電子カルテに記録してください
#     - 現病歴
#     - 既往歴
#     - 診断
#     - 症状
#     - 治療計画
#     - 家族歴
#     - 生活歴
        
# 出力のフォーマットは以下のようなkey-valueペア形式で記述してください。
# 各keyは情報のカテゴリー(例: 診断、症状、家族歴、既往歴)を示し、各 value は「-」で始まる具体的な情報のリストとして記述します。

#     フォーマットは以下のようです:
#         Key1:
#         - value1
#         - value2

#         key2:
#         - value3
#         - value4
        
#     {five_shot_examples}
    
# ただし、以下の制約に注意してください:
# 1. 各keyは以下のkey_listに限定されており、**必ずリスト内のものを使用してください**。リストにないkeyは使用しないでください。
    
#     key_list = [
#         '現病歴',
#         '既往歴',
#         '診断',
#         '症状',
#         '治療計画',
#         '家族歴',
#         '生活歴',
#     ]
    
#     ただし、全てのkeyが出力に含まれる必要はありません。
#     例えば、入力文から症状に関する情報しか含まれていない場合の出力は以下のようになります。
#         症状:
#         - value1
#         - value2
#     他のkeyに関する情報が含まれていないから、出力に含めないでください。
    

# 2. 入力文に関連するkeyが何も含まれていない場合、出力全体を「-」としてください。
#    例えば:
#         こんにちは。今日はどのようなご相談ですか？
#         -
#         医療情報が含まれていないため、出力は「-」としてください。
        
# 3. 入力文から医療情報が抽出できない場合は勝手な情報を出力しないでください。出力は入力文に基づいて正確な情報を記述してください。

# 4. 出力のフォーマットに従って記述してください。出力には、上記のフォーマット以外のテキストを含めないでください。

# """
