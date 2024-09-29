import gradio as gr
import pandas as pd
import re

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
    
    return output_table

def generate_mer_from_dialogue(df: gr.DataFrame):
    conv = Conversation()
    system_message = "あなたは医療アシスタントです。医師と患者の間で行われる日本語の会話を要約し、電子カルテを日本語で作成してください。"
    conv.add_message("system", system_message)
    
    for idx, row in df.iterrows():
        conv.add_message("user", row["Dialogue"])
        
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_conversation_and_summarize_into_electronic_medical_record",
            "description": """この機能は、医師と患者または患者の家族との会話を読み取り、重要な情報を抽出し、標準的な形式で電子カルテを作成します。""",
            "parameters": {
                "type": "object",
                "properties": {
                    "presenting_complaint": {
                        "type": "array",
                        "description": "患者が訴えている主な健康上の問題や、診察を求める理由のリスト。複数の訴えが含まれる場合があります。",
                        "items": {
                            "type": "string",
                            "description": "主訴。",
                        },
                    },
                    "symptoms": {  
                        "type": "array",
                        "description": "患者が報告する症状のリスト。複数の症状を含む場合があります。",
                        "items": {
                            "type": "string",
                            "description": "症状。",
                        },
                    },
                    "physical_examination_findings": {
                        "type": "array",
                        "description": "身体検査の結果（例：血圧、体温、脈拍など）や、臓器や身体機能の評価結果。",
                        "items": {
                            "type": "string",
                            "description": "身体検査の所見。",
                        },
                    },
                    "test_results": {
                        "type": "array",
                        "description": "血液検査、尿検査、画像検査（例：X線、CT、MRIなど）の結果。",
                        "items": {
                            "type": "string",
                            "description": "検査結果。",
                        },
                    },
                    "diagnosis": {
                        "type": "string",
                        "description": "医師による診断。病名や症状名を含む。",
                    },
                    "treatment_plan": {
                        "type": "array",
                        "description": "医師が処方した治療計画。薬物治療、手術、リハビリテーション、生活指導などが含まれる場合があります。",
                        "items": {
                            "type": "string",
                            "description": "治療計画の詳細。",
                        },
                    },
                    "prescription_info": {
                        "type": "array",
                        "description": "処方された薬の詳細。薬剤名、投与量、投与方法、投与期間などを含む。",
                        "items": {
                            "type": "string",
                            "description": "処方された薬の詳細。",
                        },
                    },
                    "follow_up": {
                        "type": "array",
                        "description": "フォローアップの計画に関する情報。次回の診察予定や、患者の経過や状態の変化に関する情報を含む。",
                        "items": {
                            "type": "string",
                            "description": "フォローアップの詳細。",
                        },
                    },
                },
                "required": ["presenting_complaint", "symptoms", "diagnosis", "treatment_plan"],  
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
        