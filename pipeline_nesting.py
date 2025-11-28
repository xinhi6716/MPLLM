import json
import concurrent.futures
from core.tracker import CostTracker
from core.nano import nano
from prompts.templates import MPLLMPrompts

# [NEW] 匯入 Nesting 工具箱 (負責平行運算與微型 Prompt)
import utils.nesting_manager as nesting

# === Helper Functions (保持與 pipeline_core 一致) ===
def parse_json_garbage(text):
    """強健的 JSON 解析器，能處理 markdown 與雜訊"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start != -1 and end != -1:
            try:
                return json.loads(clean[start:end])
            except:
                pass
    return {}

def nesting_run_wrapper(persona, user_text, model_fn):
    """
    適配器 (Adapter): 
    讓 nesting_manager 可以呼叫 core.nano
    """
    return nano(persona, user_text, model_fn)

# === Nesting Pipeline ===
def run_pipeline(task_type: str, item_data: dict, models: dict, tracker: CostTracker):
    """
    MPLLM Nesting Pipeline (實驗組)
    策略: 
    1. Switch (同 Baseline)
    2. Layer 1: Code-Level Parallelism (Researcher + Thinker) via Manager
    3. Layer 2: Sequential Logic (MiniMux) via Manager
    """
    trace = {}
    print(f"\n🧪 [Nesting Pipeline] Start: {task_type}")

    # ==========================================
    # Phase 1: Switch (Persona Generation)
    # ==========================================
    print(f"🔹 [1/Switch] Generating Personas...")
    
    if task_type == 'trivia':
        sys_p = MPLLMPrompts.TRIVIA_SWITCH_SYSTEM
        user_p = MPLLMPrompts.TRIVIA_SWITCH_USER.format(**item_data)
    elif task_type == 'codenames':
        sys_p = MPLLMPrompts.CODENAMES_SWITCH_SYSTEM
        wl = item_data.get('word_list', [])
        tw = item_data.get('target_words', [])
        user_p = MPLLMPrompts.CODENAMES_SWITCH_USER.format(
            n=len(tw), target_words=", ".join(tw), word_list=", ".join(wl)
        )
    elif task_type == 'logic':
        sys_p = MPLLMPrompts.LOGIC_SWITCH_SYSTEM
        user_p = MPLLMPrompts.LOGIC_SWITCH_USER.format(**item_data)
    else:
        raise ValueError(f"Unknown task: {task_type}")

    switch_raw, t1, _ = nano(sys_p, user_p, models['mini'])
    tracker.add('switch', t1, 0)
    trace['switch_raw'] = switch_raw
    
    switch_data = parse_json_garbage(switch_raw)
    personas_data = {"groups": switch_data.get('groups', [])}
    
    # Fallback if switch fails
    if not personas_data["groups"]:
        print("⚠️ Switch failed, using fallback groups.")
        personas_data["groups"] = [{"g": i+1, "r": "Expert", "t": "Thinker"} for i in range(3)]

    # ==========================================
    # Phase 2 & 3: Nesting Layer 1 (Parallel Execution)
    # ==========================================
    # 這裡取代了 pipeline_core 的 Phase 2 (Combined Researcher) 和 Phase 3 (Loop Thinker)
    print(f"🔹 [2+3/Layer1] Parallel Researchers & Thinkers...")
    
    final_output = {}

    if task_type == 'trivia':
        # 使用 ThreadPool 同時啟動 "Researcher群組" 和 "Thinker群組"
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            
            # Task A: Researchers (呼叫 Manager 的平行邏輯)
            future_r = executor.submit(
                nesting.run_trivia_researcher_nested,
                model_fn=models['nano'],
                nano_run_fn=nesting_run_wrapper, # 傳入適配器
                personas_data=personas_data,
                questions=item_data.get('questions', [])
            )
            
            # Task B: Thinkers (呼叫 Manager 的平行邏輯)
            future_t = executor.submit(
                nesting.run_trivia_thinker_parallel,
                model_fn=models['nano'],
                nano_run_fn=nesting_run_wrapper, # 傳入適配器
                personas_data=personas_data,
                topic=item_data.get('topic', '')
            )
            
            # 等待並獲取結果
            r_answers_list, toks_r = future_r.result()
            thinker_results, toks_t = future_t.result()
            
        tracker.add('researcher_nested', toks_r, 0)
        tracker.add('thinker_nested', toks_t, 0)

        # ==========================================
        # Phase 4: Nesting Layer 2 (MiniMux)
        # ==========================================
        print(f"🔹 [4/Minimux] Mini Model Arbitrating & Writing...")
        
        final_json_str, toks_mux = nesting.run_trivia_minimux_nested(
            mini_model_fn=models['mini'],
            mini_run_fn=nesting_run_wrapper,
            topic=item_data.get('topic', ''),
            questions=item_data.get('questions', []),
            r_answers=r_answers_list,
            t_results=thinker_results
        )
        tracker.add('minimux_nested', toks_mux, 0)
        
        final_output = parse_json_garbage(final_json_str)

    elif task_type == 'codenames':
        # TODO: Codenames Nesting logic implementation
        print("⚠️ Codenames Nesting not implemented in manager yet. Returning empty.")
        final_output = {}

    elif task_type == 'logic':
        # TODO: Logic Nesting logic implementation
        print("⚠️ Logic Nesting not implemented in manager yet. Returning empty.")
        final_output = {}

    return final_output, trace