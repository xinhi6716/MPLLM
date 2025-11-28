import json
import re
import concurrent.futures
from typing import List, Dict, Any
from prompts.templates import MPLLMPrompts

"""
MPLLM Nesting Manager (Mechanism Layer) - Fixed Names
-----------------------------------------------------
修正重點：
1. 函數名稱改回與 pipeline_nesting.py 一致
   - run_researchers -> run_trivia_researcher_nested
   - run_thinkers    -> run_trivia_thinker_parallel
   - run_minimux_nested -> run_trivia_minimux_nested
"""

# ==============================================================================
# SECTION 1: 微型提示詞 (Micro-Prompts)
# ==============================================================================
TRIVIA_MICRO_RESEARCHER = """### SYSTEM: Specialist Researcher
Role: {persona}
Task: Answer strictly. If unknown, say "unknown". Keep answers short.
Questions:
{questions_text}
Output JSON: {{"answers": ["ans1", "ans2", ...]}}"""

MINI_SELECTOR_PROMPT = """### SYSTEM: Fact Arbitrator
Task: Review answers, Verify facts, Correct errors, Finalize output.
Topic: {topic}
Data:
{formatted_data}
Output JSON: {{"final_answers": ["a1", "a2", ...], "correction_log": "..."}}"""

MINI_WRITER_PROMPT = """### SYSTEM: Storyteller
Task: Write a story ({topic}) using Facts + Advice.
Facts: {final_answers}
Advice: {thinker_directions}
Output JSON: {{"final_story": "..."}}"""

# ==============================================================================
# SECTION 2: 工具函數 (Helpers)
# ==============================================================================
def robust_json_parser(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text).replace("```json", "").replace("```", "").strip()
    if "{" in text:
        text = text[text.find("{"):text.rfind("}")+1]
    try: return json.loads(text)
    except: return {}

def format_trivia_data(questions, r_answers):
    formatted = ""
    for i, q in enumerate(questions):
        a1 = r_answers[0][i] if len(r_answers)>0 and len(r_answers[0])>i else "unk"
        a2 = r_answers[1][i] if len(r_answers)>1 and len(r_answers[1])>i else "unk"
        a3 = r_answers[2][i] if len(r_answers)>2 and len(r_answers[2])>i else "unk"
        formatted += f"Q{i+1}: {q}\n R1:{a1}|R2:{a2}|R3:{a3}\n"
    return formatted

# ==============================================================================
# SECTION 3: 平行運算邏輯 (Parallel Logic)
# ==============================================================================

# --- [A] Researcher Logic ---
def _worker_researcher(idx, persona, questions_text, model_fn, nano_run_fn):
    prompt = TRIVIA_MICRO_RESEARCHER.format(persona=persona, questions_text=questions_text)
    res, tokens, _ = nano_run_fn(persona=persona, user_text=prompt, model_fn=model_fn)
    
    data = robust_json_parser(res)
    answers = data.get("answers", [])
    
    q_count = questions_text.count("\n") + 1
    if len(answers) < q_count: answers.extend(["unknown"] * (q_count - len(answers)))
    return idx, answers, tokens

# [FIX] 改名為 run_trivia_researcher_nested 以匹配 pipeline_nesting.py
def run_trivia_researcher_nested(model_fn, nano_run_fn, personas_data, questions):
    """
    同時啟動 3 個 Researcher 執行緒
    """
    groups = personas_data.get("groups", [])
    q_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    r_answers = [[], [], []]
    total_tokens = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, g in enumerate(groups):
            futures.append(executor.submit(_worker_researcher, i, g.get('r','Expert'), q_text, model_fn, nano_run_fn))
            
        for f in concurrent.futures.as_completed(futures):
            idx, ans, t = f.result()
            if 0 <= idx < 3: r_answers[idx] = ans
            total_tokens += t
    return r_answers, total_tokens

# --- [B] Thinker Logic ---
def _worker_thinker(idx, persona, topic, model_fn, nano_run_fn):
    prompt = MPLLMPrompts.TRIVIA_THINKER.format(thinker_persona=persona, topic=topic)
    res, tokens, _ = nano_run_fn(persona=persona, user_text=prompt, model_fn=model_fn)
    data = robust_json_parser(res)
    direction = data.get("creative_direction") or res
    return idx, {"analysis": {"creative_direction": direction}}, tokens

# [FIX] 改名為 run_trivia_thinker_parallel 以匹配 pipeline_nesting.py
def run_trivia_thinker_parallel(model_fn, nano_run_fn, personas_data, topic):
    """
    同時啟動 3 個 Thinker 執行緒
    """
    groups = personas_data.get("groups", [])
    t_results = {}
    total_tokens = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, g in enumerate(groups):
            # Thinker 的 key 使用 group_id (1, 2, 3)
            futures.append(executor.submit(_worker_thinker, i+1, g.get('t','Thinker'), topic, model_fn, nano_run_fn))
            
        for f in concurrent.futures.as_completed(futures):
            idx, res, t = f.result()
            t_results[idx] = res
            total_tokens += t
    return t_results, total_tokens

# --- [C] MiniMux Logic (Sequential) ---
# [FIX] 改名為 run_trivia_minimux_nested 以匹配 pipeline_nesting.py
def run_trivia_minimux_nested(mini_model_fn, mini_run_fn, topic, questions, r_answers, t_results):
    """
    Step 1: Arbitrate -> Step 2: Write
    """
    total_tokens = 0
    
    # Step 1: Arbitrate
    p1 = MINI_SELECTOR_PROMPT.format(topic=topic, formatted_data=format_trivia_data(questions, r_answers))
    res1, t1, _ = mini_run_fn(persona="Arbitrator", user_text=p1, model_fn=mini_model_fn)
    total_tokens += t1
    final_ans = robust_json_parser(res1).get("final_answers", r_answers[0] if r_answers else [])

    # Step 2: Write
    dirs = " | ".join([v.get('analysis',{}).get('creative_direction','') for k,v in t_results.items()])
    p2 = MINI_WRITER_PROMPT.format(topic=topic, final_answers=json.dumps(final_ans), thinker_directions=dirs)
    res2, t2, _ = mini_run_fn(persona="Writer", user_text=p2, model_fn=mini_model_fn)
    total_tokens += t2
    
    final_story = robust_json_parser(res2).get("final_story", "Failed")
    correction = robust_json_parser(res1).get("correction_log", "None")
    
    return json.dumps({"final_answers": final_ans, "final_story": final_story, "correction_log": correction}), total_tokens