import json
import re
import concurrent.futures
from core.tracker import CostTracker
from core.nano import nano
from prompts.templates import MPLLMPrompts

# === Helper Functions ===
def parse_json_garbage(text):
    """強健的 JSON 解析器，能處理 markdown 與雜訊"""
    try:
        # 嘗試直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 清理 markdown
        clean = text.replace("```json", "").replace("```", "").strip()
        # 尋找最外層的 {}
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start != -1 and end != -1:
            try:
                return json.loads(clean[start:end])
            except:
                pass
    return {}

def get_persona_name(p_data):
    """從 dict 或 str 中提取角色名稱"""
    if isinstance(p_data, dict):
        return p_data.get('persona', 'Expert')
    return str(p_data) if p_data else "Expert"

# === Core Pipeline ===
def run_mpllm_pipeline(task_type: str, item_data: dict, models: dict, tracker: CostTracker):
    """
    MPLLM 通用流水線 (整合版)
    支援 User 自定義的 Combined Researcher 策略
    """
    trace = {}
    print(f"\n🚀 Starting MPLLM Pipeline for Task: {task_type}")

    # ==========================================
    # Phase 1: Switch (Persona Generation)
    # ==========================================
    print(f"🔹 [1/Switch] Generating Personas...")
    
    if task_type == 'trivia':
        sys_p = MPLLMPrompts.TRIVIA_SWITCH_SYSTEM
        user_p = MPLLMPrompts.TRIVIA_SWITCH_USER.format(**item_data)
    elif task_type == 'codenames':
        sys_p = MPLLMPrompts.CODENAMES_SWITCH_SYSTEM
        # Codenames 需要 word_list 字串
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
    groups = switch_data.get('groups', [])
    
    # Fallback if switch fails
    if not groups:
        print("⚠️ Switch failed, using fallback groups.")
        groups = [{"g": i+1, "r": "Expert", "t": "Thinker"} for i in range(3)]

    # ==========================================
    # Phase 2: Researcher (Combined Strategy)
    # ==========================================
    print(f"🔹 [2/Researcher] Gathering Facts (Combined Strategy)...")
    
    researcher_results = {} # 格式: {group_id: context_string/dict}
    
    # 提取角色名稱
    r_names = [get_persona_name(g.get('researcher')) for g in groups]
    # 補滿 3 個
    while len(r_names) < 3: r_names.append("Expert")

    if task_type == 'trivia':
        # Trivia 使用 Combined Researcher Prompt
        q_list = item_data.get('questions', [])
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(q_list)])
        
        r_prompt = MPLLMPrompts.TRIVIA_RESEARCHER.format(
            r1=r_names[0], r2=r_names[1], r3=r_names[2],
            questions=q_str
        )
        r_out, t2, _ = nano("", r_prompt, models['mini']) # Researcher 通常用 Mini
        tracker.add('researcher', t2, 0)
        
        # 解析 Researcher JSON: {"r1": [...], "r2": [...], ...}
        r_json = parse_json_garbage(r_out)
        researcher_results = {
            1: json.dumps(r_json.get('r1', [])),
            2: json.dumps(r_json.get('r2', [])),
            3: json.dumps(r_json.get('r3', []))
        }
        trace['researcher_raw'] = r_out

    elif task_type == 'codenames':
        # Codenames 使用 Combined Researcher Prompt
        tw = item_data.get('target_words', [])
        wl = item_data.get('word_list', [])
        others = [w for w in wl if w not in tw]
        
        r_prompt = MPLLMPrompts.CODENAMES_RESEARCHER.format(
            r1_persona=r_names[0], r2_persona=r_names[1], r3_persona=r_names[2],
            clue=switch_data.get('initial_clue', 'unknown'),
            target_words=", ".join(tw),
            others=", ".join(others)
        )
        r_out, t2, _ = nano("", r_prompt, models['mini'])
        tracker.add('researcher', t2, 0)
        
        r_json = parse_json_garbage(r_out)
        researcher_results = {
            1: r_json.get('r1', {}),
            2: r_json.get('r2', {}),
            3: r_json.get('r3', {})
        }
        trace['researcher_raw'] = r_out

    else:
        # Logic 或其他任務可能不需要 Researcher 或使用個別模式
        pass

    # ==========================================
    # Phase 3: Thinker (Parallel)
    # ==========================================
    print(f"🔹 [3/Thinker] Parallel Reasoning...")
    
    thinker_results = {}
    
    def run_single_thinker(g_idx, group):
        t_persona = get_persona_name(group.get('thinker'))
        
        if task_type == 'trivia':
            # Trivia Thinker 只需要 Topic 和 Persona，不需要 Researcher 的詳細答案
            # (根據 method.py 的邏輯，Thinker 給 Creative Direction，Minimux 才看 Facts)
            t_prompt = MPLLMPrompts.TRIVIA_THINKER.format(
                thinker_persona=t_persona,
                topic=item_data.get('topic', '')
            )
        
        elif task_type == 'codenames':
            # Codenames Thinker 給 Advice
            t_prompt = MPLLMPrompts.CODENAMES_THINKER.format(
                thinker_persona=t_persona,
                clue=switch_data.get('initial_clue', ''),
                target_words=", ".join(item_data.get('target_words', []))
            )
            
        elif task_type == 'logic':
            # Logic Thinker 還是維持原樣，傳入題目
            t_prompt = f"Role: {t_persona}\nTask: Solve this logic puzzle.\nInput: {item_data.get('inputs')}"
            
        else:
            t_prompt = f"Role: {t_persona}\nTask: {str(item_data)}"

        # 呼叫 Nano
        out, tok, _ = nano("", t_prompt, models['nano']) # Thinker 用 Nano/High Model
        return g_idx, out, tok

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_single_thinker, i+1, g) for i, g in enumerate(groups)]
        for future in concurrent.futures.as_completed(futures):
            gid, out, tok = future.result()
            tracker.add(f'thinker_g{gid}', tok, 0)
            thinker_results[gid] = parse_json_garbage(out) # 嘗試解析 JSON

    trace['thinker_results'] = thinker_results

    # ==========================================
    # Phase 4: Minimux (Decision)
    # ==========================================
    print(f"🔹 [4/Minimux] Aggregating & Final Decision...")
    
    final_output = {}
    
    if task_type == 'trivia':
        # 準備資料給 Minimux
        # Researcher 答案 + Thinker 方向
        r1_val = researcher_results.get(1, "[]")
        r2_val = researcher_results.get(2, "[]")
        r3_val = researcher_results.get(3, "[]")
        
        t1_val = thinker_results.get(1, {}).get('creative_direction', '')
        t2_val = thinker_results.get(2, {}).get('creative_direction', '')
        t3_val = thinker_results.get(3, {}).get('creative_direction', '')
        
        mini_prompt = MPLLMPrompts.TRIVIA_MINIMUX.format(
            topic=item_data.get('topic', ''),
            questions=json.dumps(item_data.get('questions', [])),
            researcher1=r1_val, researcher2=r2_val, researcher3=r3_val,
            thinker1=t1_val, thinker2=t2_val, thinker3=t3_val
        )
        
        mini_out, t4, _ = nano("", mini_prompt, models['mini'])
        tracker.add('minimux', t4, 0)
        final_output = parse_json_garbage(mini_out)

    elif task_type == 'codenames':
        # 準備 Minimux 資料
        # 格式化 Researcher results (Risk/Coverage)
        r_str_list = []
        for gid in range(1, 4):
            r_res = researcher_results.get(gid, {})
            p_name = get_persona_name(groups[gid-1].get('researcher'))
            r_str_list.append(f"Group {gid} ({p_name}): risk={r_res.get('risk')}, coverage={r_res.get('coverage')}")
            
        # 格式化 Thinker advice
        t_str_list = []
        for gid in range(1, 4):
            advice = thinker_results.get(gid, {}).get('advice', 'none')
            t_str_list.append(f"Group {gid}: {advice}")
            
        mini_prompt = MPLLMPrompts.CODENAMES_MINIMUX.format(
            target_words=", ".join(item_data.get('target_words', [])),
            word_list=", ".join(item_data.get('word_list', [])),
            clue_switch=switch_data.get('initial_clue', ''),
            researcher_result="\n".join(r_str_list),
            thinker_result="\n".join(t_str_list)
        )
        
        mini_out, t4, _ = nano("", mini_prompt, models['mini'])
        tracker.add('minimux', t4, 0)
        final_output = parse_json_garbage(mini_out)
        
        # Codenames 特有的 Phase 5: Guesser
        print(f"🔹 [5/Guesser] Generating Guesses...")
        final_clue = final_output.get('final_clue', switch_data.get('initial_clue'))
        
        guesser_prompt = MPLLMPrompts.CODENAMES_GUESSER.format(
            clue=final_clue,
            board=", ".join(item_data.get('word_list', [])),
            k=len(item_data.get('target_words', []))
        )
        
        guess_out, t5, _ = nano("", guesser_prompt, models['nano']) # Guesser 用好一點的模型
        tracker.add('guesser', t5, 0)
        final_output['guessed_words'] = guess_out # 簡單回傳字串，之後再解析

    elif task_type == 'logic':
        # Logic 的簡單聚合
        decider_p = get_persona_name(switch_data.get('decider'))
        candidates = "\n".join([f"G{k}: {v}" for k, v in thinker_results.items()])
        mini_prompt = f"Synthesize these solutions:\n{candidates}"
        
        mini_out, t4, _ = nano(decider_p, mini_prompt, models['nano'])
        tracker.add('minimux', t4, 0)
        final_output = {'final_answer': mini_out}

    return final_output, trace