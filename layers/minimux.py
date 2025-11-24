# layers/minimux.py
from core.nano import nano
from prompts.templates import MPLLMPrompts

def run_minimux_layer(candidates: list, model_fn):
    """整合多個思考結果"""
    # 將候選答案組合成字串
    candidates_text = ""
    for i, cand in enumerate(candidates):
        candidates_text += f"\n=== Candidate {i+1} ===\n{cand}\n"
    
    output, tokens, msgs = nano(
        persona=MPLLMPrompts.MINIMUX_SYSTEM,
        prompt=f"Here are the candidates to evaluate:\n{candidates_text}",
        model_fn=model_fn
    )
    return output, tokens