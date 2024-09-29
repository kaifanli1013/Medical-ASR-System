import os
import ast
import concurrent
import json
import os
import pandas as pd
import tiktoken
from csv import writer
from IPython.display import display, Markdown, Latex
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from termcolor import colored

GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"
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
    
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response

def calculate_embeddings_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """为每个对话行计算嵌入并添加到DataFrame中."""
    df['embedding'] = df['Dialogue'].apply(lambda x: embedding_request(x).data[0].embedding)
    # print("Generated embeddings:", df['embedding'])
    return df

def summarize_conversation(content):
    system_message = "あなたは医療アシスタントです。医師と患者の間で行われる日本語の会話を要約し、電子カルテを日本語で作成してください。"
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

# en version
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "read_conversation_and_summarize_into_electronic_medical_record",
#             "description": """This function reads a conversation between a doctor and a patient or the patient's family, 
#             extracts key information, and generates an electronic medical record in standard written format.""",
#             # "strict": True,
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "presenting_complaint": {
#                         "type": "array",
#                         "description": "A list of the patient's main health concerns or reasons for seeking medical attention, which may include multiple complaints.",
#                         "items": {
#                             "type": "string",
#                             "description": "A presenting complaint.",
#                         },
#                     },
#                     "symptoms": {  
#                         "type": "array",
#                         "description": "A list of symptoms reported by the patient, which may include multiple distinct symptoms.",
#                         "items": {
#                             "type": "string",
#                             "description": "A symptom.",
#                         },
#                     },
#                     "physical_examination_findings": {
#                         "type": "array",
#                         "description": "Results from the physical examination, including vital signs (e.g., blood pressure, temperature, pulse) and evaluations of organ and body functions.",
#                         "items": {
#                             "type": "string",
#                             "description": "A specific finding from the physical examination.",
#                         },
#                     },
#                     "test_results": {
#                         "type": "array",
#                         "description": "Results from diagnostic tests such as blood tests, urine tests, or imaging (e.g., X-rays, CT, MRI).",
#                         "items": {
#                             "type": "string",
#                             "description": "A specific test result.",
#                         },
#                     },
#                     "diagnosis": {
#                         "type": "string",
#                         "description": "The diagnosis made by the doctor, including the name of the disease or condition.",
#                     },
#                     "treatment_plan": {
#                         "type": "array",
#                         "description": "The treatment plan prescribed by the doctor, which may include medications, surgeries, rehabilitation, or lifestyle advice.",
#                         "items": {
#                             "type": "string",
#                             "description": "A specific treatment recommendation or action plan.",
#                         },
#                     },
#                     "prescription_info": {
#                         "type": "array",
#                         "description": "Details of prescribed medications, including the drug name, dosage, route of administration, and duration of treatment.",
#                         "items": {
#                             "type": "string",
#                             "description": "Details of a prescribed medication.",
#                         },
#                     },
#                     "follow_up": {
#                         "type": "array",
#                         "description": "Information about follow-up plans, including the next scheduled appointment and observations of the patient's progress or changes in their condition.",
#                         "items": {
#                             "type": "string",
#                             "description": "A specific follow-up detail or plan.",
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


# Test sample: 
# system_message = "You are a medical assistant. You are talking to a patient who is describing their symptoms to you. You need to summarize the conversation into the patient's electronic medical record."
# conversation=Conversation()
# conversation.add_message("system", system_message)

# conversation.add_message("user", test_message)
# chat_response = chat_completion_request(conversation.conversation_history, tools=tools)
# print(chat_response.choices[0].message.tool_calls[0].function.arguments)

