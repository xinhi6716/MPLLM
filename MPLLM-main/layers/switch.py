# layers/switch.py
from core.nano import nano
from prompts.templates import MPLLMPrompts

def run_switch_layer(user_input: str, model_fn):
    """分析任務類型"""
    output, tokens, msgs = nano(
        persona=MPLLMPrompts.SWITCH_SYSTEM,
        prompt=f"User Request: {user_input}",
        model_fn=model_fn
    )
    return output, tokens