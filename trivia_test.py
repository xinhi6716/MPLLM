import os
import json
import random
import re
import csv
import datetime
import time
import concurrent.futures
from typing import Dict, Any, List, Tuple

from mpllm_prompts.nano import nano_run, mini_mux
from mpllm_prompts.switch_prompts import trivia_personas_switch_prompt
from mpllm_prompts.prompts import (
    trivia_researcher_prompt,
    trivia_thinker_prompt,
    trivia_minimux_prompt,
)

# API 設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
MINI_MODEL_NAME = os.environ.get("MINI_MODEL_NAME", "gpt-4.1-mini")

# OpenAI 客戶端設定
try:
    from openai import OpenAI
    _use_openai_v1 = True
except Exception:
    _use_openai_v1 = False
    import openai


def build_openai_model_fn(model: str):
    """建立 OpenAI 模型函數"""
    if _use_openai_v1:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        def _model_fn(messages: List[Dict[str, str]]) -> Tuple[str, int]:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=5000,  # 進一步增加輸出長度限制
            )
            choice = resp.choices[0] if resp.choices else None
            text = (choice.message.content or "").strip() if choice and choice.message else ""
            
            usage = getattr(resp, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) if usage else 0)
            return text, total_tokens

        return _model_fn
    else:
        openai.api_key = OPENAI_API_KEY
        openai.api_base = OPENAI_BASE_URL

        def _model_fn(messages: List[Dict[str, str]]) -> Tuple[str, int]:
            resp = openai.ChatCompletion.create(model=model, messages=messages)
            text = (resp["choices"][0]["message"]["content"] or "").strip()
            usage = resp.get("usage") or {}
            total_tokens = int(usage.get("total_tokens") or 0)
            return text, total_tokens

        return _model_fn


def load_trivia_test_data():
    """載入 trivia 測試資料"""
    path = "./data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ===== 模塊化流程函數 =====
def run_switch_phase(model_fn, topic: str, verbose: bool = False) -> Tuple[Dict[str, Any], int]:
    """階段 1: 生成角色群組

    Returns:
        (personas_data, tokens)
        personas_data 格式: {"groups": [{"g": 1, "r": "...", "t": "..."}]}
    """
    # 動態生成角色組合
    switch_prompt = trivia_personas_switch_prompt.replace("{topic}", topic)
    
    # 建立限制 output tokens 的 model_fn（用於 Switch）
    def limited_switch_model_fn(messages):
        if _use_openai_v1:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=900,  # Switch 只需要簡短的 JSON（約 50-150 tokens）
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) if usage else 0)
            return text, total_tokens
        else:
            import openai
            openai.api_key = OPENAI_API_KEY
            openai.api_base = OPENAI_BASE_URL
            resp = openai.ChatCompletion.create(
                model=MINI_MODEL_NAME,
                messages=messages,
                max_tokens=900  # Switch 只需要簡短的 JSON（約 50-150 tokens）
            )
            text = (resp["choices"][0]["message"]["content"] or "").strip()
            usage = resp.get("usage") or {}
            total_tokens = int(usage.get("total_tokens", 0))
            return text, total_tokens
    
    switch_result, switch_tokens, _ = nano_run(persona="", user_text=switch_prompt, model_fn=limited_switch_model_fn)
    
    # 解析角色
    try:
        personas_data = json.loads(switch_result)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", switch_result, flags=re.S)
        if m:
            try:
                personas_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                # 使用備用角色組合
                personas_data = {
                    "groups": [
                        {"g": 1, "r": "Fact Expert", "t": "Story Weaver"},
                        {"g": 2, "r": "Data Scholar", "t": "Tale Scribe"},
                        {"g": 3, "r": "Knowledge Keeper", "t": "Myth Builder"},
                    ]
                }
        else:
            # 使用備用角色組合
            personas_data = {
                "groups": [
                    {"g": 1, "r": "Fact Expert", "t": "Story Weaver"},
                    {"g": 2, "r": "Data Scholar", "t": "Tale Scribe"},
                    {"g": 3, "r": "Knowledge Keeper", "t": "Myth Builder"},
                ]
            }


    # 顯示生成的角色群組（僅在 verbose 模式下）
    if verbose:
        print("步驟 1: 生成角色群組")
        for group in personas_data.get("groups", []):
            group_id = group.get("g", 0)
            researcher = group.get("r", "Unknown")
            thinker = group.get("t", "Unknown")
            print(f"   群組{group_id}: {researcher} | {thinker}")
    
    return personas_data, switch_tokens


def get_role_name(role):
    """從角色數據中提取字符串名稱，處理 LLM 可能返回的列表格式"""
    if isinstance(role, str):
        return role
    elif isinstance(role, list) and len(role) > 0:
        return str(role[0])
    else:
        return str(role) if role else "Expert"


def run_researcher_thinker_phase(
    model_fn,
    personas_data: Dict[str, Any],
    topic: str,
    questions: List[str],
    verbose: bool = False,
) -> Tuple[List[List[str]], Dict[int, Dict], int]:
    """階段 2: 並行執行 Researcher (合併) + Thinker

    Returns:
        (researcher_answers_list, thinker_results, total_tokens)
        researcher_answers_list: 3組答案列表 [["ans1", "ans2", ...], [...], [...]]
        thinker_results: {group_id: {'persona': ..., 'analysis': {...}, 'tokens': ...}}
    """
    groups = personas_data.get("groups", [])

    # 準備合併版 Researcher prompt（一次調用得到3組答案）
    r1 = get_role_name(groups[0].get("r", "Expert1"))
    r2 = get_role_name(groups[1].get("r", "Expert2"))
    r3 = get_role_name(groups[2].get("r", "Expert3"))
    
    # 格式化問題列表為易讀格式
    questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    researcher_prompt = (
        trivia_researcher_prompt
        .replace("{r1}", r1)
        .replace("{r2}", r2)
        .replace("{r3}", r3)
        .replace("{questions}", questions_formatted)
    )

    # 準備 Thinker prompts
    thinker_prompt = trivia_thinker_prompt.replace("{topic}", topic)

    # 建立限制 output tokens 的 model_fn（用於 Thinker）
    def limited_thinker_model_fn(messages: List[Dict[str, str]]) -> Tuple[str, int]:
        if _use_openai_v1:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=1500,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) if usage else 0)
            return text, total_tokens
        else:
            openai.api_key = OPENAI_API_KEY
            openai.api_base = OPENAI_BASE_URL
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1500,
            )
            text = (resp["choices"][0]["message"]["content"] or "").strip()
            usage = resp.get("usage") or {}
            total_tokens = int(usage.get("total_tokens") or 0)
            return text, total_tokens

    def run_researchers_combined():
        """一次調用執行所有3個researcher"""
        try:
            result, tokens, messages = nano_run(persona="", user_text=researcher_prompt, model_fn=model_fn)
            
            # 如果輸出為空，返回錯誤標記
            if not result or len(result) == 0:
                return "researchers", 0, "combined", '{"r1":[],"r2":[],"r3":[]}', tokens
            return "researchers", 0, "combined", result, tokens
        except Exception as e:
            # 返回空結果
            return "researchers", 0, "combined", '{"r1":[],"r2":[],"r3":[]}', 0

    def run_thinker(group_id, thinker_persona):
        personalized_prompt = thinker_prompt.replace("{thinker_persona}", thinker_persona)
        try:
            result, tokens, _ = nano_run(persona="", user_text=personalized_prompt, model_fn=limited_thinker_model_fn)
        except Exception as e:
            result, tokens = "", 0
        return "thinker", group_id, thinker_persona, result, tokens

    # 並行執行
    phase_results: Dict[str, Dict[str, Any]] = {}
    total_tokens = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        # 提交合併版 Researcher 任務（只1個）
        researcher_future = executor.submit(run_researchers_combined)
        futures.append((researcher_future, "researchers", 0))

        # 提交 Thinker 任務（3個）
        for i, group in enumerate(groups):
            thinker_persona = get_role_name(group.get("t", "Writer"))
            future = executor.submit(run_thinker, i + 1, thinker_persona)
            futures.append((future, "thinker", i + 1))

        # 收集結果
        for future in concurrent.futures.as_completed([f[0] for f in futures]):
            for task_future, task_type, group_id in futures:
                if future == task_future:
                    try:
                        task_type, group_id, persona, result, tokens = future.result()
                        total_tokens += tokens

                        if task_type == "researchers":
                            # 處理合併版 researcher 結果
                            phase_results["researchers_combined"] = {
                                "persona": "Combined Researchers",
                                "result": result,
                                "tokens": tokens,
                            }
                        else:
                            # 處理 thinker 結果
                            phase_results[f"thinker_{group_id}"] = {
                                "persona": persona,
                                "result": result,
                                "tokens": tokens,
                            }
                    except Exception as e:
                        if task_type == "researchers":
                            phase_results["researchers_combined"] = {
                                "persona": "Error",
                                "result": '{"r1":[],"r2":[],"r3":[]}',
                                "tokens": 0,
                                "error": str(e),
                            }
                        else:
                            phase_results[f"thinker_{group_id}"] = {
                                "persona": "Error",
                                "result": '{"creative_direction": "error"}',
                                "tokens": 0,
                                "error": str(e),
                            }
                    break

    # 提取 Researcher 答案（從合併結果中）
    researcher_answers_list: List[List[str]] = [[], [], []]
    if "researchers_combined" in phase_results:
        raw_result = phase_results["researchers_combined"]["result"]
        
        try:
            combined_data = json.loads(raw_result)
            researcher_answers_list[0] = combined_data.get("r1", [])
            researcher_answers_list[1] = combined_data.get("r2", [])
            researcher_answers_list[2] = combined_data.get("r3", [])
            
        except json.JSONDecodeError as e:
            # 嘗試用正則表達式提取
            m = re.search(r"\{[\s\S]*?\}", raw_result)
            if m:
                try:
                    combined_data = json.loads(m.group(0))
                    researcher_answers_list[0] = combined_data.get("r1", [])
                    researcher_answers_list[1] = combined_data.get("r2", [])
                    researcher_answers_list[2] = combined_data.get("r3", [])
                    
                except json.JSONDecodeError:
                    pass

    # 提取 Thinker 結果
    thinker_results: Dict[int, Dict[str, Any]] = {}
    for i, _group in enumerate(groups):
        group_id = i + 1
        thinker_key = f"thinker_{group_id}"
        if thinker_key in phase_results:
            t_result = phase_results[thinker_key]
            
            # 先處理輸出
            raw_result = t_result["result"].strip()
            
            # 移除雙大括號
            if raw_result.startswith("{{") and raw_result.endswith("}}"):
                raw_result = raw_result[1:-1]
            
            try:
                analysis = json.loads(raw_result)
            except json.JSONDecodeError:
                # 處理多種可能的格式
                cleaned = raw_result.strip()
                
                # 1. 嘗試移除外層雙大括號
                if cleaned.startswith("{{") and cleaned.endswith("}}"):
                    cleaned = cleaned[1:-1]
                
                # 2. 如果有多行，嘗試提取最後一行的 JSON
                if "\n" in cleaned:
                    lines = cleaned.split("\n")
                    for line in reversed(lines):  # 從後往前找
                        line = line.strip()
                        if line.startswith("{"):
                            cleaned = line
                            break
                
                # 3. 移除第二層雙大括號（如果存在）
                if cleaned.startswith("{{") and cleaned.endswith("}}"):
                    cleaned = cleaned[1:-1]
                
                try:
                    analysis = json.loads(cleaned)
                except json.JSONDecodeError:
                    # 4. 最後用正則表達式提取
                    m = re.search(r"\{[^{}]*creative_direction[^{}]*\}", cleaned)
                    if m:
                        try:
                            analysis = json.loads(m.group(0))
                        except json.JSONDecodeError:
                            analysis = {"creative_direction": "error parsing"}
                    else:
                        analysis = {"creative_direction": "error parsing"}

            thinker_results[group_id] = {
                "persona": t_result["persona"],
                "analysis": analysis,
                "tokens": t_result["tokens"],
            }

    return researcher_answers_list, thinker_results, total_tokens


def run_decider_phase(
    model_fn,
    personas_data: Dict[str, Any],
    topic: str,
    researcher_answers_list: List[List[str]],
    thinker_results: Dict[int, Dict],
    verbose: bool = False,
) -> Tuple[Dict[int, Dict], int]:
    """階段 3: 並行執行 Decider（已廢棄，不再使用）

    Returns:
        (decider_results, total_tokens)
        decider_results: {group_id: {'decider_persona': ..., 'thinker_persona': ..., 'final_story': "...", 'tokens': ...}}
    
    Note: 此函數已不再使用，保留僅供參考。如需恢復 Decider 階段，需重新 import trivia_decider_prompt。
    """
    # 此函數已廢棄，直接返回空結果
    # 如需恢復 Decider 階段，請參考 git 歷史記錄中的完整實現，並重新 import trivia_decider_prompt
    return {}, 0


def run_minimux_phase(mini_model_fn, topic, questions, researcher_answers_list, thinker_results, verbose=False):
    """階段 3: Mini Mux 評比（直接使用 Thinker 分析，不使用 Decider）"""
    q_json  = json.dumps(questions, ensure_ascii=False, separators=(',',':'))
    r1_json = json.dumps(researcher_answers_list[0] if len(researcher_answers_list) > 0 else [], ensure_ascii=False, separators=(',',':'))
    r2_json = json.dumps(researcher_answers_list[1] if len(researcher_answers_list) > 1 else [], ensure_ascii=False, separators=(',',':'))
    r3_json = json.dumps(researcher_answers_list[2] if len(researcher_answers_list) > 2 else [], ensure_ascii=False, separators=(',',':'))

    def clip_text(s: str, max_words=20):
        """裁切文字到指定字數"""
        ws = re.findall(r"\S+", s or "")
        return " ".join(ws[:max_words])

    # 使用 Thinker 的 creative_direction 而不是 Decider 故事
    t1 = clip_text(thinker_results.get(1,{}).get('analysis',{}).get('creative_direction',''), 20)
    t2 = clip_text(thinker_results.get(2,{}).get('analysis',{}).get('creative_direction',''), 20)
    t3 = clip_text(thinker_results.get(3,{}).get('analysis',{}).get('creative_direction',''), 20)

    # 執行 Mini Mux（傳入 topic 和 Thinker 分析）
    minimux_result, minimux_tokens, _ = mini_mux(trivia_minimux_prompt, mini_model_fn, 
                                            topic=topic, questions=q_json, 
                                            researcher1=r1_json, R1=r1_json,
                                            researcher2=r2_json, R2=r2_json,
                                            researcher3=r3_json, R3=r3_json,
                                            thinker1=t1, thinker2=t2, thinker3=t3)

    # 解析 Mini Mux 結果
    try:
        minimux_data = json.loads(minimux_result)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*?\}", minimux_result)
        if m:
            try:
                minimux_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                minimux_data = {"final_answers": [], "final_story": "解析失敗"}
        else:
            minimux_data = {"final_answers": [], "final_story": "解析失敗"}


    return minimux_data, minimux_tokens


# ===== 測試函數 =====
def normalize_answer(ans: str) -> str:
    """正規化答案"""
    if not ans:
        return ""
    s = str(ans).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # 移除標點符號
    s = re.sub(r"\b(the|a|an|and|or|of|in|on|at|to|for|with|by)\b", " ", s)  # 移除停用詞
    s = re.sub(r"\s+", " ", s).strip()  # 正規化空格
    return s


def test_trivia():
    """單題完整測試（使用模塊化流程）"""
    print("=== TRIVIA 單題測試 ===")
    print("=" * 50)

    # 載入測試資料
    test_data = load_trivia_test_data()
    if not test_data:
        print("無法載入測試資料")
        return

    # 隨機選一題測試
    item = random.choice(test_data)
    topic = item.get("topic", "General")
    questions = item.get("questions", [])
    correct_answers_list = item.get("answers", [])

    print(f"主題: {topic}")
    print(f"問題數: {len(questions)}")
    print()

    # 設定模型
    model_fn = build_openai_model_fn(MODEL_NAME)
    mini_model_fn = build_openai_model_fn(MINI_MODEL_NAME)

    # 記錄開始時間
    start_time = time.time()

    # 執行四個階段
    personas_data, switch_tokens = run_switch_phase(model_fn, topic, verbose=True)
    print()

    print("步驟 2: 並行執行 Researcher + Thinker", end="", flush=True)
    researcher_answers_list, thinker_results, rt_tokens = run_researcher_thinker_phase(
        model_fn, personas_data, topic, questions, verbose=True
    )
    print(" ✓")

    print("步驟 3: Mini Mux 評比", end="", flush=True)
    minimux_data, minimux_tokens = run_minimux_phase(
        mini_model_fn, topic, questions, researcher_answers_list, thinker_results, verbose=False
    )
    print(" ✓\n")

    # 記錄結束時間
    elapsed_time = time.time() - start_time

    # 總結
    print("=" * 50)
    total_tokens = switch_tokens + rt_tokens + minimux_tokens
    print(f"\n⏱️  執行時間: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分鐘)")
    
    # 答案比對
    final_answers = minimux_data.get("final_answers", [])
    correct_count = 0
    total_questions_count = len(questions)
    
    print(f"\n✅ 答案比對:")
    for i, final_ans in enumerate(final_answers):
        if i >= len(correct_answers_list):
            break
        
        final_norm = normalize_answer(final_ans)
        if not final_norm:
            status = "❌"
            continue
        
        # 檢查是否匹配任何正確答案
        matched = False
        for correct_ans in correct_answers_list[i]:
            correct_norm = normalize_answer(correct_ans)
            
            # 精確/包含匹配
            if (
                final_norm == correct_norm
                or final_norm in correct_norm
                or correct_norm in final_norm
            ):
                correct_count += 1
                matched = True
                status = "✓"
                break
            
            # 關鍵詞匹配（長度>2的詞）
            final_words = [w for w in final_norm.split() if len(w) > 2]
            correct_words = [w for w in correct_norm.split() if len(w) > 2]
            if any(w in correct_norm for w in final_words) or any(w in final_norm for w in correct_words):
                correct_count += 1
                matched = True
                status = "✓"
                break
        
        if not matched:
            status = "✗"
        
        print(f"   問題 {i+1}: {status} | 預測: {final_ans[:30]:<30} | 正確: {', '.join(correct_answers_list[i][:2])}")
    
    # 顯示準確率
    accuracy = (correct_count / total_questions_count * 100) if total_questions_count > 0 else 0
    print(f"\n🎯 準確率: {correct_count}/{total_questions_count} ({accuracy:.1f}%)")
    
    # 保存單題結果到 CSV（使用統一檔案）
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "results_trivia.csv")
    
    # 確保 CSV 有標頭（若檔案不存在、為空或缺標頭則補上）
    header = ["時間", "主題", "問題數", "正確數", "準確率", "總 Token", "Switch", "Researcher+Thinker", "Mini Mux", "執行時間(秒)", "測試模式", "Researcher答案", "Thinker分析", "Mini Mux答案", "Mini Mux故事"]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(header)
    else:
        try:
            # 檢查檔案是否有正確的 header
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                
            # 如果沒有 header 或 header 不正確，寫入新 header
            if not existing_header or "時間" not in existing_header or len(existing_header) != len(header):
                with open(csv_path, 'r+', encoding='utf-8-sig') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(",".join(header) + "\n" + content)
        except Exception:
            # 如果讀取失敗，直接建立新檔案
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(header)

    # 獲取當前時間戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 準備各階段輸出內容
    r_output = json.dumps(researcher_answers_list, ensure_ascii=False, separators=(',', ':'))
    t_output = json.dumps([thinker_results[i]["analysis"] for i in sorted(thinker_results.keys())], ensure_ascii=False, separators=(',', ':'))
    minimux_answers = json.dumps(minimux_data.get("final_answers", []), ensure_ascii=False, separators=(',', ':'))
    minimux_story = minimux_data.get("final_story", "")

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, topic, len(questions), correct_count, f"{accuracy:.1f}%",
            total_tokens, switch_tokens, rt_tokens, minimux_tokens, f"{elapsed_time:.2f}", "single",
            r_output, t_output, minimux_answers, minimux_story
        ])
    
    print(f"\n📊 結果已儲存至: {csv_path}")
    print("=" * 50)


def test_trivia_batch():
    """批次測試所有 100 題（使用模塊化流程）"""
    # 讀取題庫
    data = load_trivia_test_data()
    total_groups = len(data)

    model_fn = build_openai_model_fn(MODEL_NAME)
    mini_model_fn = build_openai_model_fn(MINI_MODEL_NAME)

    # 統計變數
    total_tokens = 0
    nano_tokens_total = 0
    mini_tokens_total = 0
    correct_count = 0
    total_questions = 0
    error_count = 0

    # 各階段 token 統計
    total_switch_tokens = 0
    total_rt_tokens = 0
    total_minimux_tokens = 0
    
    # 時間統計
    total_elapsed_time = 0.0

    # 確保 results 資料夾存在
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 使用統一的 CSV 檔案
    csv_path = os.path.join(results_dir, "results_trivia.csv")

    # 檢查並寫入 CSV header（只在最開始做一次）
    header = ["時間", "主題", "問題數", "正確數", "準確率", "總 Token", "Switch", "Researcher+Thinker", "Mini Mux", "執行時間(秒)", "測試模式", "Researcher答案", "Thinker分析", "Mini Mux答案", "Mini Mux故事"]
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    else:
        # 檢查檔案是否有正確的 header
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                
            # 如果沒有 header 或 header 不正確，寫入新 header
            if not existing_header or "時間" not in existing_header or len(existing_header) != len(header):
                with open(csv_path, 'r+', encoding='utf-8-sig') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(",".join(header) + "\n" + content)
        except Exception:
            # 如果讀取失敗，直接建立新檔案
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    for idx, item in enumerate(data):
        topic = item.get("topic", "General")
        questions = item.get("questions", [])
        correct_answers_list = item.get("answers", [])

        # 計算即時準確度
        current_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0

        # 進度條顯示
        percent = (idx / total_groups)
        filled = int(30 * percent)
        bar = "=" * filled + "-" * (30 - filled)
        print(
            f"\r進度: |{bar}| {idx}/{total_groups} ({percent*100:.1f}%) | 準確度: {current_accuracy:.1f}% | {topic[:20]:<20}",
            end="",
            flush=True,
        )

        # 記錄用的變數
        item_switch_tokens = 0
        item_rt_tokens = 0
        item_minimux_tokens = 0
        item_error = ""
        item_group_correct = 0
        item_error_occurred = False
        
        # 記錄各階段輸出內容
        item_researcher_output = []
        item_thinker_output = []
        item_minimux_answers = []
        item_minimux_story = ""
        
        # 記錄單題開始時間
        item_start_time = time.time()

        try:
            # 執行四個階段
            personas_data, switch_tokens = run_switch_phase(model_fn, topic, verbose=False)
            item_switch_tokens = switch_tokens
            total_tokens += switch_tokens
            nano_tokens_total += switch_tokens
            total_switch_tokens += switch_tokens

            researcher_answers_list, thinker_results, rt_tokens = run_researcher_thinker_phase(
                model_fn, personas_data, topic, questions, verbose=False
            )
            item_rt_tokens = rt_tokens
            total_tokens += rt_tokens
            nano_tokens_total += rt_tokens
            total_rt_tokens += rt_tokens

            minimux_data, minimux_tokens = run_minimux_phase(
                mini_model_fn, topic, questions, researcher_answers_list, thinker_results, verbose=False
            )
            item_minimux_tokens = minimux_tokens
            total_tokens += minimux_tokens
            mini_tokens_total += minimux_tokens
            total_minimux_tokens += minimux_tokens

            # 計算準確度（使用 Mini Mux 的 final_answers）
            final_answers = minimux_data.get("final_answers", [])
            group_correct = 0
            for i, final_ans in enumerate(final_answers):
                if i >= len(correct_answers_list):
                    break

                final_norm = normalize_answer(final_ans)
                if not final_norm:
                    continue

                # 檢查是否匹配任何正確答案
                matched = False
                for correct_ans in correct_answers_list[i]:
                    correct_norm = normalize_answer(correct_ans)

                    # 精確/包含匹配
                    if (
                        final_norm == correct_norm
                        or final_norm in correct_norm
                        or correct_norm in final_norm
                    ):
                        group_correct += 1
                        matched = True
                        break

                    # 關鍵詞匹配（長度>2的詞）
                    final_words = [w for w in final_norm.split() if len(w) > 2]
                    correct_words = [w for w in correct_norm.split() if len(w) > 2]
                    if any(w in correct_norm for w in final_words) or any(w in final_norm for w in correct_words):
                        group_correct += 1
                        matched = True
                        break

                if not matched:
                    # 未匹配則不加分
                    pass

            item_group_correct = group_correct
            correct_count += group_correct
            total_questions += len(questions)
            
            # 記錄各階段輸出內容
            item_researcher_output = researcher_answers_list
            item_thinker_output = [thinker_results[i]["analysis"] for i in sorted(thinker_results.keys())]
            item_minimux_answers = minimux_data.get("final_answers", [])
            item_minimux_story = minimux_data.get("final_story", "")

        except Exception as e:
            error_count += 1
            item_error_occurred = True
            item_error = str(e)
        
        # 記錄單題結束時間
        item_elapsed_time = time.time() - item_start_time
        total_elapsed_time += item_elapsed_time
        
        # 立即寫入每一題的結果到 CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item_accuracy = (item_group_correct / len(questions) * 100) if questions else 0
        item_total_tokens = item_switch_tokens + item_rt_tokens + item_minimux_tokens
        
        # 準備各階段輸出內容
        r_output = json.dumps(item_researcher_output, ensure_ascii=False, separators=(',', ':'))
        t_output = json.dumps(item_thinker_output, ensure_ascii=False, separators=(',', ':'))
        minimux_answers = json.dumps(item_minimux_answers, ensure_ascii=False, separators=(',', ':'))
        minimux_story = item_minimux_story
        
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, topic, len(questions), item_group_correct, f"{item_accuracy:.1f}%",
                item_total_tokens, item_switch_tokens, item_rt_tokens, item_minimux_tokens, f"{item_elapsed_time:.2f}", "batch",
                r_output, t_output, minimux_answers, minimux_story
            ])

    # 完成進度條
    bar = "=" * 30
    final_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    print(f"\r進度: |{bar}| {total_groups}/{total_groups} (100.0%) | 準確度: {final_accuracy:.1f}% | 完成!{' '*20}")
    
    # 最終統計
    print(f"\n{'='*60}")
    print("測試完成！")
    print(f"總題組: {total_groups} | 成功: {total_groups - error_count} | 錯誤: {error_count}")
    print(f"總問題數: {total_questions}")
    print(f"正確答案: {correct_count}")
    print(f"準確度: {final_accuracy:.1f}%")
    print(f"總 Token: {total_tokens:,} | 平均/組: {total_tokens/total_groups:.0f}")
    
    # 時間統計
    avg_time_per_group = total_elapsed_time / total_groups if total_groups > 0 else 0
    print(f"總執行時間: {total_elapsed_time:.2f}秒 ({total_elapsed_time/60:.2f}分鐘)")
    print(f"平均/組: {avg_time_per_group:.2f}秒 ({avg_time_per_group/60:.2f}分鐘)")
    
    # 寫入批次測試總體統計到 CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = total_elapsed_time
    
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, f"批次測試({total_groups}組)", total_questions, correct_count, f"{final_accuracy:.1f}%",
            total_tokens, total_switch_tokens, total_rt_tokens, total_minimux_tokens, f"{total_time:.2f}", "batch_summary",
            "", "", "", ""  # 批次總統計不包含個別輸出內容
        ])
    
    print(f"\n📊 詳細結果已儲存至: {csv_path}")
    print(f"{'='*60}")


def main():
    """主程式入口"""
    print("請選擇測試模式:")
    print("1. single - 單題測試")
    print("2. batch - 批次測試（100題）\n")

    mode = input("請輸入 'single' 或 'batch': ").strip().lower()

    if mode == "batch":
        test_trivia_batch()
    elif mode == "single":
        test_trivia()
    else:
        print("無效選項，執行單題測試...")
        test_trivia()


if __name__ == "__main__":
    main()
