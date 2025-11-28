# pipeline_core.py
import json
import concurrent.futures
from core.tracker import CostTracker
from prompts.templates import MPLLMPrompts
from layers import switch, researcher, thinker, minimux, guesser
from core.nano import nano

def run_mpllm_pipeline(task_type: str, item_data: dict, models: dict, tracker: CostTracker):
    """
    MPLLM 通用流水線：支援 Trivia, Codenames, Logic
    """
    trace = {}
    
    # ==========================================
    # Layer 1: Switch (Persona Generation)
    # ==========================================
    print(f"🔹 [1/Switch] Generating Personas for {task_type}...")
    
    # 1. 準備 Prompt
    if task_type == 'trivia':
        sys_prompt = MPLLMPrompts.TRIVIA_SWITCH_SYSTEM
        user_prompt = MPLLMPrompts.TRIVIA_SWITCH_USER.format(**item_data)
    elif task_type == 'codenames':
        sys_prompt = MPLLMPrompts.CODENAMES_SWITCH_SYSTEM
        user_prompt = MPLLMPrompts.CODENAMES_SWITCH_USER.format(**item_data)
    elif task_type == 'logic':
        sys_prompt = MPLLMPrompts.LOGIC_SWITCH_SYSTEM
        # 邏輯題目有時候很長，我們只取前段避免 token 爆炸，或確保完整傳入
        # 這裡假設 item_data['inputs'] 是字串
        user_prompt = MPLLMPrompts.LOGIC_SWITCH_USER.format(**item_data)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 2. 執行 Switch
    switch_raw, t1, _ = nano(sys_prompt, user_prompt, models['mini'])
    tracker.add('gpt-4o-mini', t1, 0)
    trace['switch_raw'] = switch_raw

    # 3. 解析 JSON (增強版)
    switch_data = {}
    try:
        # 嘗試清理 markdown
        clean_json = switch_raw.replace("```json", "").replace("```", "").strip()
        start = clean_json.find('{')
        end = clean_json.rfind('}') + 1
        if start != -1 and end != -1:
            clean_json = clean_json[start:end]
        
        switch_data = json.loads(clean_json)
    except (json.JSONDecodeError, ValueError):
        print(f"⚠️ Switch JSON Parsing Failed! Raw output:\n{switch_raw[:200]}...")
        # 不直接 return error，而是進入 Fallback 流程

    # ==========================================
    # Fallback Mechanism (救援機制)
    # ==========================================
    groups = switch_data.get('groups', [])
    
    if not groups:
        print(f"⚠️ Warning: Switch layer produced 0 groups. Using DEFAULT personas for {task_type}.")
        # 根據任務類型提供預設的 Persona，確保流程不中斷
        if task_type == 'trivia':
            groups = [
                {"researcher": "You are a creative researcher.", "thinker": "You are a storyteller."},
                {"researcher": "You are a fact checker.", "thinker": "You are a plot designer."},
                {"researcher": "You are a historian.", "thinker": "You are a narrative expert."}
            ]
        elif task_type == 'codenames':
            groups = [
                {"thinker": "You are a linguistic expert specializing in word associations."},
                {"thinker": "You are a lateral thinking puzzle solver."},
                {"thinker": "You are a cryptic clue analyzer."}
            ]
        elif task_type == 'logic':
            groups = [
                {"thinker": "You are a deductive logic expert. Focus on elimination."},
                {"thinker": "You are a constraint analyst. Focus on mapping relationships."},
                {"thinker": "You are a matrix puzzle solver. Focus on grid consistency."}
            ]
        
        # 更新 switch_data 以便後續使用
        switch_data['groups'] = groups
        switch_data['decider'] = "You are a master synthesizer."

    # ==========================================
    # Layer 2 & 3: Collaborative Groups (Parallel)
    # ==========================================
    print(f"🔹 [2-3/Collab] Executing {len(groups)} Persona Groups...")
    
    group_results = []
    
    def process_group(group, g_idx):
        # 取得 Persona (處理巢狀或字串)
        r_p = group.get('researcher')
        if isinstance(r_p, dict): r_p = r_p.get('persona', str(r_p))
        
        t_p = group.get('thinker')
        if isinstance(t_p, dict): t_p = t_p.get('persona', str(t_p))
        
        # --- Layer 2: Researcher ---
        context = ""
        if task_type == 'trivia':
            topic = item_data.get('topic', '')
            questions = item_data.get('questions', [])
            q_str = "\n".join([f"- {q}" for q in questions])
            
            research_prompt = f"Topic: {topic}\nQuestions: {q_str}\nProvide facts."
            # 若沒有 researcher persona (如 logic/codenames)，跳過
            if r_p:
                r_out, tok_r, _ = nano(str(r_p), research_prompt, models['mini'])
                context = r_out
        
        # --- Layer 3: Thinker ---
        # 根據不同任務格式化題目
        if task_type == 'codenames':
            task_prompt = f"""
            Task: Play Codenames as Spymaster.
            Target Words: {item_data.get('target_words')}
            Bad Words: {item_data.get('word_list')}
            Generate a clue word and number.
            """
        elif task_type == 'logic':
            task_prompt = f"""
            Task: Solve this Logic Puzzle step-by-step.
            Puzzle: {item_data.get('inputs')}
            """
        elif task_type == 'trivia':
            q_list = item_data.get('questions', [])
            q_str = "\n".join([f"- {q}" for q in q_list])
            task_prompt = f"Write a story about {item_data.get('topic')} answering:\n{q_str}"
        else:
            task_prompt = str(item_data)

        # 執行 Thinker (注意：有些任務只有 thinker 沒有 researcher)
        t_out, tok_t, _ = thinker.run_single_thinker(context, task_prompt, g_idx, str(t_p), models['nano'])
        
        return t_out

    # 並行執行
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_group, g, i) for i, g in enumerate(groups)]
        for future in concurrent.futures.as_completed(futures):
            try:
                group_results.append(future.result())
            except Exception as e:
                print(f"⚠️ Group execution error: {e}")
                
    trace['group_results'] = group_results

    # ==========================================
    # Layer 4 & 5: Decider / MiniMux
    # ==========================================
    print("🔹 [4/Decider] Aggregating results...")
    
    decider_p = switch_data.get('decider')
    if isinstance(decider_p, dict): decider_p = decider_p.get('persona', 'You are the decider.')
    
    candidates_text = "\n".join([f"Group {i+1} Opinion: {res}" for i, res in enumerate(group_results)])
    
    if task_type == 'trivia':
        q_list = item_data.get('questions', [])
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(q_list)])
        instruction = MPLLMPrompts.TRIVIA_WRITER.format(
            topic=item_data.get('topic', ''), n=len(q_list), questions=q_str
        )
        final_prompt = f"{instruction}\n\n=== Expert Opinions ===\n{candidates_text}\n\nCreate the final story."
    
    elif task_type == 'logic':
        # 【關鍵修改】這裡強制要求格式，與 evaluate_logic 對接
        final_prompt = f"""
        You are the Logic Arbiter.
        Here are expert deductions:
        {candidates_text}
        
        Your Goal: Identify the ONE correct option (1, 2, 3, 4, or 5).
        
        CRITICAL OUTPUT FORMAT:
        Explain your reasoning briefly, but your LAST LINE must be exactly:
        Final Answer: X
        (Replace X with the single digit of the correct option).
        """
    
    elif task_type == 'codenames':
        final_prompt = f"""
        Review these clues:
        {candidates_text}
        Target words: {item_data.get('target_words')}
        
        Output the best single-word clue and count.
        """
    else:
        final_prompt = f"Synthesize the final answer from these opinions:\n{candidates_text}"
    
    final_answer, t_final, _ = nano(str(decider_p), final_prompt, models['nano'])
    tracker.add('gpt-4o-mini', t_final, 0)
    trace['final_answer'] = final_answer

    return final_answer, trace