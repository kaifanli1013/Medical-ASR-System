import requests
import pysrt
import os
import pandas as pd
import gradio as gr

class SIP3API:
    def __init__(self, base_url):
        
        self.base_url = base_url
        self.endpoints = {
            # 医療単語を入力として、単語に該当するSIP3辞書の項目を返します。
            # 辞書項目には、正規形、難易度、外部データベースへのリンク先、信頼度などが含まれます。
            "entry": "entry", 
            
            # 医療テキストを入力として、テキストに含まれる全てのエンティティを返します。
            # エンティティには、テキスト中の出現位置、カテゴリ、該当する辞書項目などが含まれます。
            "entities": "entities",
            
            # 医療テキストを入力として、テキストに含まれる全てのエンティティの正規形を返します。
            # オプションで、返却されるエンティティのカテゴリを指定できます。
            "norms": "norms",
            
            # 医療テキストを入力として、テキストに含まれる全てのエンティティのコード（外部データベースへの紐づけ先）を返します。
            # オプションで、返却されるコードを指定できます。
            "codes": "codes",
            
            # 医療テキストを入力として、テキストに含まれる全てのエンティティの難易度の低い語（患者向け語）を返します。
            # オプションで、難易度の閾値を指定できます。
            "patient_terms": "patient_terms",
        }
        
    def api_call(self, endpoint, params):
        if endpoint not in self.endpoints:
            raise ValueError("Invalid endpoint")
        
        url = self.base_url + self.endpoints[endpoint]
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print("Error Response status:", response.status_code)
            print("Error Message:", response.text)
            return None            
        
    # def standardize_subtitle_file(self,
    #                              files_subtitles: list,
    #                              progress=gr.Progress()) -> list:
    #     """
    #     Standardize subtitle file from generated subtitles

    #     Parameters
    #     ----------
    #     files_subtitles: list
    #         List of generated subtitle files from gr.Files()
    #     progress: gr.Progress
    #         Indicator to show progress directly in gradio.

    #     Returns
    #     ----------
    #     standardized_result_str:
    #         Result of standardization to return to gr.Textbox()
    #     standardized_result_file_path:
    #         Output file path to return to gr.Files()
    #     """
    #     try:
    #         files_info = {}
    #         for file in files_subtitles:
    #             subtitle_path = file.name
    #             subs = pysrt.open(subtitle_path)

    #             standardized_texts = []
    #             for sub in subs:
    #                 params = {
    #                     "text": sub.text,
    #                     "category": "disease",
    #                     # "min_reliability": "C"
    #                 }
    #                 # print(f"Calling SIP3 API with params: {params}")
    #                 try:
    #                     standardized_entities = self.api_call("norms", params)
    #                     # print(f"Received response: {standardized_entities}")
    #                 except Exception as api_error:
    #                     # print(f"API call error: {api_error}")
    #                     return [f"API call error: {api_error}", None]

    #                 if not standardized_entities:
    #                     # print(f"API returned no entities for text: {sub.text}")
    #                     standardized_texts.append(sub.text)
    #                     continue
                    
    #                 # 初始化 standardized_text
    #                 standardized_text = sub.text
                    
    #                 # 替换文本中的实体为标准化名称
    #                 for entity in standardized_entities:
    #                     standardized_text = standardized_text.replace(entity['text'], entity['standard_name'][0])
                    
    #                 standardized_texts.append(standardized_text)

    #             result_texts = []
    #             for sub, standardized_text in zip(subs, standardized_texts):
    #                 result_texts.append(f"{sub.index}\n{sub.start} --> {sub.end}\n{standardized_text}\n")

    #             new_subtitle_path = subtitle_path.replace(".srt", "_standardized.srt")
    #             with open(new_subtitle_path, 'w', encoding='utf-8') as f:
    #                 f.writelines(result_texts)

    #             files_info[os.path.basename(subtitle_path)] = {
    #                 "standardized_texts": "\n".join(result_texts),
    #                 "path": new_subtitle_path
    #             }

    #         total_result = ''
    #         for file_name, info in files_info.items():
    #             total_result += '------------------------------------\n'
    #             total_result += f'{file_name}\n\n'
    #             total_result += f'{info["standardized_texts"]}'

    #         standardized_result_str = f"Standardization completed.\n\n{total_result}"
    #         standardized_result_file_path = [info['path'] for info in files_info.values()]

    #         print(f'{standardized_result_str=}')
    #         print(f'{standardized_result_file_path=}')
            
    #         return [standardized_result_str, standardized_result_file_path]

    #     except Exception as e:
    #         print(f"Error standardizing file: {e}")

    def standardize_subtitle_file(self,
                                 df: gr.DataFrame,
                                 progress=gr.Progress()) -> list:
        """
        Standardize subtitle file from generated subtitles
        """
        
        try:    
            # print(df_list)
            # for df in df_list:
            for index, row in df.iterrows():
                if row['EMR'] == "":
                    continue
                params = {
                    "text": row['EMR'],
                    "category": "disease",
                    "min_reliability": "E"
                }
                # print("row['EMR']:", row['EMR'])
                # print(f"Calling SIP3 API with params: {params}")
                try:
                    standardized_entities = self.api_call("norms", params)
                    # print(f"Received response: {standardized_entities}")
                except Exception as api_error:
                    # print(f"API call error: {api_error}")
                    return [f"API call error: {api_error}", None]
                
                if not standardized_entities:
                    # print(f"API returned no entities for text: {sub.text}")
                    continue
                
                # print(f"Standardizing entities: {standardized_entities}")
                
                # 对每个标准化实体进行替换
                dialogue_text = df.at[index, 'EMR']  # 取出对话
                for entity in standardized_entities:
                    # 执行替换
                    # print(f"Dialogue before replacement: {dialogue_text}")
                    # print(f"Replacing entity: {entity['text']} with {entity['standard_name'][0]}")
                    dialogue_text = dialogue_text.replace(entity['text'], entity['standard_name'][0])
                    # print(f"Dialogue after replacement: {dialogue_text}")
                    
                # 将替换后的文本赋值回 DataFrame
                df.at[index, 'EMR'] = dialogue_text

            # print(df)
            return df
        
        except Exception as e:
            print(f"Error standardizing file: {e}")

        
if __name__ == "__main__":
    base_url = "https://nlp.sociocom.jp/sip3/api/"
    sip3_api = SIP3API(base_url)
    
    params = {
        "text": "症状: 鎖骨のがん細胞が確認された",  # 示例医学文本
        "category": "disease",  # 可选参数
        "min_reliability": "E"
    }

    # 调用/norms端点
    standardized_text = sip3_api.api_call("norms", params)
    print("标准化后的文本:", standardized_text)