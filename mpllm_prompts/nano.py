from __future__ import annotations

from typing import Callable, List, Dict, Tuple

ModelFn = Callable[[List[Dict[str, str]]], Tuple[str, int]]


def nano_build(persona: str, user_text: str) -> List[Dict[str, str]]:
    """只組合 persona 與使用者文字，回傳 messages。
    - persona: 放在 system 的完整提示（可為長版規則或角色說明）
    - user_text: 本次任務/問題（或觸發語）
    """
    return [
        {"role": "system", "content": persona},
        {"role": "user", "content": user_text},
    ]


def nano_run(persona: str, user_text: str, model_fn: ModelFn) -> Tuple[str, int, List[Dict[str, str]]]:
    """依賴注入：由外部傳入 model_fn 以執行推論。
    回傳 (out_text, tokens, messages)。
    """
    messages = nano_build(persona, user_text)
    out_text, tokens = model_fn(messages)
    return out_text, tokens, messages 


def nano(persona: str, prompt: str, model_fn: ModelFn) -> Tuple[str, int, List[Dict[str, str]]]:
    """依賴注入最小實作：
    - nano() 只負責把 persona + prompt 組成 messages
    - 實際呼叫哪個模型由外部傳入的 model_fn 決定

    回傳 (out_text, tokens, messages)
    """
    messages = nano_build(persona, prompt)
    out_text, tokens = model_fn(messages)
    return out_text, tokens, messages 


def mini_mux(minimux_prompt: str, model_fn: ModelFn, **format_kwargs) -> Tuple[str, int, List[Dict[str, str]]]:
    """Mini Mux 最終評比功能（通用版本）：
    - 接收已準備好的 prompt 模板和格式化參數
    - 使用指定模型（通常是 gpt-5-mini）進行最終評比
    - 適用於不同任務（trivia, codenames, logic 等）
    
    Args:
        minimux_prompt: mini mux 的 prompt 模板（包含 {placeholder}）
        model_fn: 模型函數（應該使用 gpt-5-mini 或其他強模型）
        **format_kwargs: 用於格式化 prompt 的任意關鍵字參數
    
    Returns:
        (result_text, tokens, messages)
    
    Examples:
        # Trivia 任務
        mini_mux(prompt, model_fn, 
                 questions="...", researcher1="...", decider1="...", ...)
        
        # Codenames 任務  
        mini_mux(prompt, model_fn,
                 hint_word="...", guesses1="...", guesses2="...", ...)
    """
    # 格式化 prompt - 使用 replace 而不是 format 避免雙大括號問題
    user_text = minimux_prompt
    for key, value in format_kwargs.items():
        user_text = user_text.replace(f"{{{key}}}", str(value))
    
    messages = nano_build("", user_text)  # 沒有 persona，直接使用 user_text
    out_text, tokens = model_fn(messages)
    return out_text, tokens, messages