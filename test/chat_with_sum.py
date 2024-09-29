import gradio as gr
import pandas as pd
import re
import json

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored


GPT_MODEL = "gpt-4o"
# EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI()

# chat completion request
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
# conversation completion request
class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

def parse_srt_to_table(content):
    pattern = r"(SPEAKER_\d+\|.+)"
    matches = re.findall(pattern, content)

    data = []
    for match in matches:
        speaker, dialogue = match.split('|', 1)
        data.append([speaker.strip(), dialogue.strip(), ""])  # EMR列为空
        
    df = pd.DataFrame(data, columns=["Speaker", "Dialogue", "EMR"])
    return df

def parse_and_summarize(file):
    # open the file
    with open(file.name, 'r', encoding='utf-8') as srt_file:
        content = srt_file.read()
        
    output_table = parse_srt_to_table(content)
    df = generate_mer_from_dialogue(output_table)
    
    return output_table

def generate_mer_from_dialogue(df: gr.DataFrame):
    conv = Conversation()
    system_message = "あなたは医療アシスタントです。医師と患者の間で行われる日本語の会話を要約し、電子カルテを日本語で作成してください。"
    conv.add_message("system", system_message)
    
    for idx, row in df.iterrows():
        conv.add_message("user", row["Dialogue"])
        
        response = chat_completion_request(
            messages=conv.conversation_history,
            tools=tools,
        )
        
        tool_call_id = response.choices[0].message.tool_calls[0].id
        tool_function_name = response.choices[0].message.tool_calls[0].function.name
        content = response.choices[0].message.tool_calls[0].function.arguments
        
        conv.conversation_history.append(
            {
                "role": "assistant",
                "tool_call_id": tool_call_id,
                "tool_function_name": tool_function_name,
                "content": content,
            }
        )
        
        content_json = json.loads(content)
        new_entries = []
        # update the EMR column
        for key, value in content_json.items():
            if key not in ("previous_summary", "current_sentence") and value:
                if not isinstance(value, list):
                    new_entries.append(f"{key}: {value}")
                else:
                    new_entries.append(f"{key}: {', '.join(value)}")
        
        df.at[idx, "EMR"] = "\n".join(new_entries)
        
    return df

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "read_conversation_and_summarize_into_electronic_medical_record",
#             "description": """この機能は、医師と患者または患者の家族との会話を読み取り、重要な情報を抽出し、標準的な形式で電子カルテを作成します。""",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "presenting_complaint": {
#                         "type": "array",
#                         "description": "患者が訴えている主な健康上の問題や、診察を求める理由のリスト。複数の訴えが含まれる場合があります。",
#                         "items": {
#                             "type": "string",
#                             "description": "主訴。",
#                         },
#                     },
#                     "symptoms": {  
#                         "type": "array",
#                         "description": "患者が報告する症状のリスト。複数の症状を含む場合があります。",
#                         "items": {
#                             "type": "string",
#                             "description": "症状。",
#                         },
#                     },
#                     "physical_examination_findings": {
#                         "type": "array",
#                         "description": "身体検査の結果（例：血圧、体温、脈拍など）や、臓器や身体機能の評価結果。",
#                         "items": {
#                             "type": "string",
#                             "description": "身体検査の所見。",
#                         },
#                     },
#                     "test_results": {
#                         "type": "array",
#                         "description": "血液検査、尿検査、画像検査（例：X線、CT、MRIなど）の結果。",
#                         "items": {
#                             "type": "string",
#                             "description": "検査結果。",
#                         },
#                     },
#                     "diagnosis": {
#                         "type": "string",
#                         "description": "医師による診断。病名や症状名を含む。",
#                     },
#                     "treatment_plan": {
#                         "type": "array",
#                         "description": "医師が処方した治療計画。薬物治療、手術、リハビリテーション、生活指導などが含まれる場合があります。",
#                         "items": {
#                             "type": "string",
#                             "description": "治療計画の詳細。",
#                         },
#                     },
#                     "prescription_info": {
#                         "type": "array",
#                         "description": "処方された薬の詳細。薬剤名、投与量、投与方法、投与期間などを含む。",
#                         "items": {
#                             "type": "string",
#                             "description": "処方された薬の詳細。",
#                         },
#                     },
#                     "follow_up": {
#                         "type": "array",
#                         "description": "フォローアップの計画に関する情報。次回の診察予定や、患者の経過や状態の変化に関する情報を含む。",
#                         "items": {
#                             "type": "string",
#                             "description": "フォローアップの詳細。",
#                         },
#                     },
#                 },
#                 "required": ["presenting_complaint", "symptoms", "diagnosis", "treatment_plan"],  
#                 "additionalProperties": False,
#             },
#         },
#     },
# ]

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_conversation_and_summarize_into_electronic_medical_record",
            "description": """
            あなたは医療アシスタントです。医師と患者、または患者の家族との対話から、以下の重要な医療情報を抽出し、電子カルテに記録してください:
            - 主訴 (presenting_complaint)
            - 症状 (symptoms)
            - 身体所見 (physical_examination_findings)
            - 検査結果 (test_results)
            - 診断 (diagnosis)
            - 治療方針 (treatment_plan)
            - 処方内容 (prescription_info)
            - フォローアップ計画 (follow_up)

            **重要な注意事項**:
            - すでに記録された情報がある場合、それを再度記録しないでください。同じ情報を異なる表現で述べていても、既に記録された情報と同じであれば記録する必要はありません。
            - すでに記録された情報は{previous_summary}に含まれています。以下に示す内容はすでに記録されています:
              - 主訴: {previous_summary['presenting_complaint']}
              - 症状: {previous_summary['symptoms']}
              - 診断: {previous_summary['diagnosis']}
              - 治療方針: {previous_summary['treatment_plan']}
            
            **新しい情報のみ**を以下の対話から抽出してください:
            {current_sentence}
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "current_sentence": {
                        "type": "string",
                        "description": "新しい対話内容。この対話から新しい医療情報を抽出します。",
                    },
                    "previous_summary": {
                        "type": "object",
                        "description": "すでに記録された電子カルテの情報。",
                        "properties": {
                            "presenting_complaint": {
                                "type": "array",
                                "description": "すでに記録された主訴のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "symptoms": {
                                "type": "array",
                                "description": "すでに記録された症状のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "physical_examination_findings": {
                                "type": "array",
                                "description": "すでに記録された身体所見のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "test_results": {
                                "type": "array",
                                "description": "すでに記録された検査結果のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "diagnosis": {
                                "type": "string",
                                "description": "すでに記録された診断内容。",
                            },
                            "treatment_plan": {
                                "type": "array",
                                "description": "すでに記録された治療計画のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "prescription_info": {
                                "type": "array",
                                "description": "すでに記録された処方内容のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "follow_up": {
                                "type": "array",
                                "description": "すでに記録されたフォローアップ計画のリスト。",
                                "items": {
                                    "type": "string"
                                }
                            },
                        }
                    },
                    "presenting_complaint": {
                        "type": "array",
                        "description": "新しく抽出された患者が訴えている主な健康上の問題や、診察を求める理由。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "symptoms": {  
                        "type": "array",
                        "description": "新しく抽出された患者が報告する症状。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "physical_examination_findings": {
                        "type": "array",
                        "description": "新しく抽出された身体検査の所見。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "test_results": {
                        "type": "array",
                        "description": "新しく抽出された検査結果。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "diagnosis": {
                        "type": "string",
                        "description": "新しく抽出された診断。",
                    },
                    "treatment_plan": {
                        "type": "array",
                        "description": "新しく抽出された治療計画。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "prescription_info": {
                        "type": "array",
                        "description": "新しく抽出された処方の詳細。",
                        "items": {
                            "type": "string"
                        },
                    },
                    "follow_up": {
                        "type": "array",
                        "description": "新しく抽出されたフォローアップの詳細。",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["previous_summary", "current_sentence", "presenting_complaint", "symptoms", "diagnosis", "treatment_plan"],  
                "additionalProperties": False,
            },
        },
    },
]

with gr.Blocks() as demo:
    gr.Markdown("Medical ASR System")
    
    with gr.Column():
        file_input = gr.File(label="Upload a subtitle file")
        generate_button = gr.Button("Generate summary")
        
    with gr.Column():
        output_table = gr.DataFrame(headers=["Speaker", "Dialogue", "EMR"], label="生成的表格")
        
    generate_button.click(parse_and_summarize, inputs=file_input, outputs=output_table)
    
if __name__ == "__main__":
    demo.launch()
        