# utils/api_client.py
import os
from openai import OpenAI
from core.nano import ModelFn

def get_openai_model_fn(model_name: str = "gpt-4o-mini", api_key: str = None) -> ModelFn:
    """
    回傳一個閉包 (Closure)，這個閉包符合 ModelFn 的簽名。
    自動處理 API key 和 Client 初始化。
    """
    # 如果沒有傳入 key，嘗試從環境變數讀取
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API Key is missing.")
        
    client = OpenAI(api_key=key)

    def model_fn(messages) -> tuple[str, int]:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )
        content = response.choices[0].message.content
        # 計算總 token (input + output)
        usage = response.usage
        total_tokens = usage.total_tokens
        # 為了 Tracker，我們這裡回傳 total，或是你可以改寫 nano 讓他回傳 usage object
        return content, total_tokens

    return model_fn