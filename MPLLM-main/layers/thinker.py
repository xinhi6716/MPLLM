# layers/thinker.py
from core.nano import nano

def run_single_thinker(context: str, question: str, idx: int, persona: str, model_fn):
    """
    執行單一次深度推理
    """
    # 若 persona 為空或 dict，給一個預設值
    if not isinstance(persona, str):
        persona = "You are a logical thinker."

    prompt = f"""
    [Context/Facts]:
    {context}
    
    [Task]:
    {question}
    
    Please provide your reasoning and answer based on your persona.
    """
    
    # 執行 nano
    output, tokens, msgs = nano(
        persona=persona, 
        prompt=prompt,
        model_fn=model_fn
    )
    
    # 關鍵修正：回傳 3 個值 (output, tokens, msgs)
    return output, tokens, msgs