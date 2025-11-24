# layers/researcher.py
from core.nano import nano
from prompts.templates import MPLLMPrompts

def run_researcher_layer(topic: str, model_fn):
    """蒐集背景知識"""
    # 這裡可以用 format 動態插入主題
    persona = MPLLMPrompts.RESEARCHER_SYSTEM.format(topic=topic)
    
    output, tokens, msgs = nano(
        persona=persona,
        prompt="Please provide the research summary now.",
        model_fn=model_fn
    )
    return output, tokens