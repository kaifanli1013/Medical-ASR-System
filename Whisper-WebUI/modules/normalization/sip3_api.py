import requests

# SIP3 API URL
sip3_url = "https://nlp.sociocom.jp/sip3/api/"

class SPI3API:
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
        
        
if __name__ == "__main__":
    base_url = "https://nlp.sociocom.jp/sip3/api/"
    sip3_api = SPI3API(base_url)
    
    params = {
        "text": "最近お腹が痛くて、時々熱もあります。",  # 示例医学文本
        "category": "disease",  # 可选参数
        "min_reliability": "C"
    }

    # 调用/norms端点
    standardized_text = sip3_api.api_call("norms", params)
    print("标准化后的文本:", standardized_text)