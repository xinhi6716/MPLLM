# layers/guesser.py
from core.nano import nano
from prompts.templates import MPLLMPrompts

def run_guesser_layer(best_solution: str, original_question: str, model_fn):
    """生成最終回覆"""
    prompt = f"""
    [Approved Solution]:
    {best_solution}
    
    [User's Question]:
    {original_question}
    """
    
    output, tokens, msgs = nano(
        persona=MPLLMPrompts.GUESSER_SYSTEM,
        prompt=prompt,
        model_fn=model_fn
    )
    return output, tokens