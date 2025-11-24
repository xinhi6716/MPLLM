# layers/thinker.py
from core.nano import nano
from prompts.templates import MPLLMPrompts

def run_single_thinker(context: str, question: str, attempt_id: int, model_fn):
    """執行單一次深度推理"""
    prompt = f"""
    [Context from Researcher]:
    {context}
    
    [Original Question]:
    {question}
    
    [Instruction]:
    This is Attempt #{attempt_id}. Please think deeply.
    """
    
    output, tokens, msgs = nano(
        persona=MPLLMPrompts.THINKER_SYSTEM,
        prompt=prompt,
        model_fn=model_fn
    )
    return output, tokens