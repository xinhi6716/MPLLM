# core/nano.py
from typing import Callable, List, Dict, Tuple

# 定義型別：ModelFn 是一個接收訊息列表，回傳 (文字, token數) 的函式
ModelFn = Callable[[List[Dict[str, str]]], Tuple[str, int]]

def nano_build(persona: str, user_text: str) -> List[Dict[str, str]]:
    """標準化訊息組裝"""
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": user_text})
    return messages

def nano(persona: str, prompt: str, model_fn: ModelFn) -> Tuple[str, int, List[Dict[str, str]]]:
    """
    MPLLM Nano 核心：
    只負責將 persona + prompt 組合成 messages，
    並呼叫外部注入的 model_fn 進行推論。
    """
    messages = nano_build(persona, prompt)
    out_text, tokens = model_fn(messages)
    return out_text, tokens, messages