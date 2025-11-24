# pipeline_core.py
import concurrent.futures
from core.tracker import CostTracker
from layers import switch, researcher, thinker, minimux, guesser

def run_mpllm_pipeline(user_input: str, models: dict, tracker: CostTracker):
    """
    執行完整的 MPLLM 流程
    models: {'mini': model_fn, 'nano': model_fn}
    """
    trace = {}
    
    # --- Step 1: Switch ---
    print("🔹 [1/5] Switch Layer analyzing...")
    switch_out, t1 = switch.run_switch_layer(user_input, models['mini'])
    tracker.add('gpt-4o-mini', t1, 0) # 簡化計算，實際可更精細
    trace['switch'] = switch_out
    
    # --- Step 2: Researcher ---
    print("🔹 [2/5] Researcher gathering info...")
    # 假設 Switch 決定需要 Research (這裡直接執行)
    research_out, t2 = researcher.run_researcher_layer(user_input, models['mini'])
    tracker.add('gpt-4o-mini', t2, 0)
    trace['research'] = research_out
    
    # --- Step 3: Thinker (Parallel) ---
    print("🔹 [3/5] Thinker Layer (x3 parallel)...")
    candidates = []
    
    def threaded_think(idx):
        # 每個執行緒都呼叫 run_single_thinker
        res, toks = thinker.run_single_thinker(research_out, user_input, idx, models['nano'])
        return res, toks

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(threaded_think, i) for i in range(1, 4)]
        for future in concurrent.futures.as_completed(futures):
            res, toks = future.result()
            candidates.append(res)
            tracker.add('gpt-4o-mini', toks, 0) # 假設 nano 也是用 mini 模擬
            
    trace['candidates'] = candidates
    
    # --- Step 4: MiniMux ---
    print("🔹 [4/5] MiniMux evaluating...")
    mux_out, t4 = minimux.run_minimux_layer(candidates, models['nano'])
    tracker.add('gpt-4o-mini', t4, 0)
    trace['minimux'] = mux_out
    
    # --- Step 5: Guesser ---
    print("🔹 [5/5] Guesser synthesizing...")
    final_out, t5 = guesser.run_guesser_layer(mux_out, user_input, models['nano'])
    tracker.add('gpt-4o-mini', t5, 0)
    trace['final_answer'] = final_out
    
    return final_out, trace