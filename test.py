import os
import json
import random
import re
import concurrent.futures
from typing import Dict, Any, List, Tuple

from mpllm_prompts.nano import nano_run, mini_mux
from mpllm_prompts.prompts import (
    trivia_researcher_prompt,
    trivia_thinker_prompt,
    trivia_decider_prompt,
    trivia_minimux_prompt,
)
from mpllm_prompts.switch_prompts import trivia_personas_switch_prompt

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-nano")
MINI_MODEL_NAME = os.environ.get("MINI_MODEL_NAME", "gpt-5-mini")

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
            resp = client.chat.completions.create(model=model, messages=messages)
            text = (resp.choices[0].message.content or "").strip()
            tokens = getattr(resp, "usage", None)
            total_tokens = getattr(tokens, "total_tokens", 0) if tokens else 0
            return text, int(total_tokens)

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
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def run_trivia_multi_group_flow(
    model_fn, topic: str, questions: List[str], personas_data: Dict[str, Any], silent: bool = False
) -> Dict[str, Any]:
    """執行 Trivia 多群組並行流程"""
    results: Dict[str, Any] = {}
    total_tokens = 0

    groups = personas_data.get("groups", [])
    
    if not silent:
        print(f"\n🔹 步驟 2：執行多群組流程 (共 {len(groups)} 組)")
        print(f"   階段 1：Researcher + Thinkers 並行...", end="", flush=True)
    
    # 準備 prompts
    questions_json = json.dumps(questions, ensure_ascii=False, separators=(",", ":"))
    researcher_prompt = trivia_researcher_prompt \
        .replace("{n}", str(len(questions))) \
        .replace("{questions}", questions_json)
    
    thinker_prompts = []
    for group in groups:
        thinker_persona = group.get("thinker", "")
        thinker_prompt = trivia_thinker_prompt \
            .replace("{thinker_persona}", thinker_persona) \
            .replace("{topic}", topic)
        thinker_prompts.append(thinker_prompt)
    
    # 並行執行 Researcher + Thinkers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        researcher_future = executor.submit(nano_run, "", researcher_prompt, model_fn)
        thinker_futures = []
        for i, thinker_prompt in enumerate(thinker_prompts):
            future = executor.submit(nano_run, "", thinker_prompt, model_fn)
            thinker_futures.append((future, i+1))
        
        results_phase1 = {}
        for future in concurrent.futures.as_completed([researcher_future] + [f[0] for f in thinker_futures]):
            if future == researcher_future:
                try:
                    r_text, r_tokens, _ = future.result()
                    results_phase1['researcher'] = (r_text, r_tokens)
                except Exception as e:
                    results_phase1['researcher_error'] = str(e)
                    results_phase1['researcher'] = ("[]", 0)
            else:
                for thinker_future, group_id in thinker_futures:
                    if future == thinker_future:
                        try:
                            t_text, t_tokens, _ = future.result()
                            results_phase1[f'thinker_{group_id}'] = (t_text, t_tokens)
                        except Exception as e:
                            results_phase1[f'thinker_{group_id}_error'] = str(e)
                            results_phase1[f'thinker_{group_id}'] = ('{"creative_direction": "error"}', 0)
                        break
    
    # 處理 Researcher 結果
    r_text, r_tokens = results_phase1['researcher']
    m = re.search(r"\[[\s\S]*?\]", r_text)
    if m:
        r_text = m.group(0)
    else:
        r_text = "[]"
    
    answers = json.loads(r_text) if r_text else []
    total_tokens += r_tokens
    
    # 處理 Thinker 結果
    thinker_results = {}
    thinker_tokens_sum = 0
    for i, group in enumerate(groups):
        group_id = i + 1
        thinker_persona = group.get("thinker", "")
        t_text, t_tokens = results_phase1[f'thinker_{group_id}']
        
        # 增強 JSON 解析容錯性
        try:
            thinker_analysis = json.loads(t_text)
        except json.JSONDecodeError:
            # 嘗試提取 JSON 對象
            m = re.search(r"\{[\s\S]*?\}", t_text)
            if m:
                try:
                    thinker_analysis = json.loads(m.group(0))
                except json.JSONDecodeError:
                    thinker_analysis = {"creative_direction": "error parsing"}
            else:
                thinker_analysis = {"creative_direction": "error parsing"}
        
        total_tokens += t_tokens
        thinker_tokens_sum += t_tokens
        thinker_results[group_id] = {
            'persona': thinker_persona,
            'analysis': thinker_analysis,
            'tokens': t_tokens
        }
    
    if not silent:
        print(f" 完成")
        print(f"   • Researcher: {r_tokens} tokens")
        print(f"   • Thinkers: {thinker_tokens_sum} tokens (G1:{thinker_results[1]['tokens']}, G2:{thinker_results[2]['tokens']}, G3:{thinker_results[3]['tokens']})")
    
    # 並行執行 Deciders
    if not silent:
        print(f"   階段 2：Deciders 並行...", end="", flush=True)
    
    answers_json = json.dumps(answers, ensure_ascii=False, separators=(',',':'))
    
    decider_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for group_id, thinker_result in thinker_results.items():
            thinker_persona = thinker_result['persona']
            thinker_analysis = thinker_result['analysis']
            analysis_json = json.dumps(thinker_analysis, ensure_ascii=False, separators=(',',':'))
            
            d_prompt = (trivia_decider_prompt
                .replace("{topic}", topic)
                .replace("{researcher_answers}", answers_json)
                .replace("{thinker_analysis}", analysis_json))
            
            future = executor.submit(nano_run, "", d_prompt, model_fn)
            decider_futures.append((future, group_id, thinker_persona))
        
        final_results = {}
        for future, group_id, thinker_persona in decider_futures:
            try:
                d_text, d_tokens, _ = future.result()
                total_tokens += d_tokens
                
                # 增強 JSON 解析容錯性
                if d_text.startswith('{{') and d_text.endswith('}}'):
                    d_text = d_text[1:-1]
                
                try:
        out = json.loads(d_text)
                except json.JSONDecodeError:
                    # 嘗試提取 JSON 對象
                    m = re.search(r"\{[\s\S]*?\}", d_text)
                    if m:
                        out = json.loads(m.group(0))
                    else:
                        raise ValueError("無法解析 JSON")
                
                final_story = out.get("final_story","").strip()
                
                final_results[group_id] = {
                    'thinker_persona': thinker_persona,
                    'final_story': final_story,
                    'tokens': d_tokens
                }
                
                    
    except Exception as e:
                final_results[group_id] = {
                    'thinker_persona': thinker_persona,
                    'final_story': "",
                    'tokens': 0,
                    'error': str(e)
                }
    
    # 計算 Decider tokens
    decider_tokens_sum = sum(r.get('tokens', 0) for r in final_results.values())
    
    if not silent:
        print(f" 完成")
        decider_tokens_list = [final_results.get(i, {}).get('tokens', 0) for i in [1, 2, 3]]
        print(f"   • Deciders: {decider_tokens_sum} tokens (G1:{decider_tokens_list[0]}, G2:{decider_tokens_list[1]}, G3:{decider_tokens_list[2]})")

    results.update({
        "answers": answers,
        "final_results": final_results,
        "tokens": total_tokens
    })
    
    return results


def test_trivia_full() -> None:
    """測試 Trivia 完整流程（單次）"""
    if not OPENAI_API_KEY:
        raise RuntimeError("請先設定環境變數 OPENAI_API_KEY。")

    # 讀取題庫
    data = load_trivia_test_data()
    idx = random.randint(0, len(data) - 1)
    item = data[idx]
    topic = item.get("topic", "General")
    questions = item.get("questions", [])
    
    print("=== TRIVIA 測試 ===")
    print(f"題組 #{idx+1} | 主題: {topic} | 問題數: {len(questions)}\n")

    # 初始化模型函式
    model_fn = build_openai_model_fn(MODEL_NAME)

    # 步驟 1：生成人格群組（使用進度條）
    print("🔹 步驟 1: 生成人格群組", end="", flush=True)
    switch_prompt = trivia_personas_switch_prompt.replace("{topic}", topic)
    switch_result, switch_tokens, _ = nano_run(persona="", user_text=switch_prompt, model_fn=model_fn)
    print(f" ✓ ({switch_tokens} tokens)")
    
    # 解析人格群組
    try:
        personas_data = json.loads(switch_result)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", switch_result, flags=re.S)
        if m:
            try:
                personas_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                personas_data = {"groups": [{"group_id": 1, "thinker": "Narrative Essayist"}]}
        else:
            personas_data = {"groups": [{"group_id": 1, "thinker": "Narrative Essayist"}]}
    
    # 顯示生成的 Thinker
    groups = personas_data.get("groups", [])
    for group in groups:
        group_id = group.get('group_id', 0)
        thinker = group.get('thinker', 'Unknown')
        print(f"   Group {group_id}'s thinker: {thinker}")
    
    # 步驟 2：多群組流程
    result = run_trivia_multi_group_flow(model_fn, topic, questions, personas_data, silent=False)

    if "error" in result:
        print("❌ 流程錯誤:", result["error"])
        return
    
    # 顯示 Researcher 答案
    answers = result.get("answers", [])
    print(f"\n📋 Researcher 答案: {answers}")
    
    # 步驟 3: Mini Mux 評比
    print(f"\n🔹 步驟 3: Mini Mux 評比", end="", flush=True)
    
    mini_model_fn = build_openai_model_fn(MINI_MODEL_NAME)
    
    # 收集 researcher 的事實答案
    facts = result.get("answers", [])
    
    # 收集三組故事
    stories = []
    final_results = result.get("final_results", {})
    
    for group_id in sorted(final_results.keys()):
        group_result = final_results[group_id]
        if "error" not in group_result:
            story = group_result.get("final_story", "")
            stories.append(story)
    else:
            stories.append("")
    
    while len(stories) < 3:
        stories.append("")
    
    # 執行 Mini Mux
    try:
        minimux_result, minimux_tokens, _ = mini_mux(
            facts, stories, trivia_minimux_prompt, mini_model_fn
        )
        
        print(f" ✓ ({minimux_tokens} tokens)")
        
        # 解析最佳組別
        try:
            minimux_data = json.loads(minimux_result)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*?\}", minimux_result)
            if m:
                try:
                    minimux_data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    minimux_data = None
        else:
                minimux_data = None
        
        if minimux_data:
            best_group = minimux_data.get("best_group", 1)
            print(f"\n🏆 最佳組別: Group {best_group}")
            
            if best_group in final_results:
                best_story = final_results[best_group].get("final_story", "")
                print(f"📖 故事: {best_story}")
    except Exception as e:
        print(f"❌ Mini Mux 錯誤: {e}")
        minimux_tokens = 0
    
    print(f"\n總 Token 消耗: {result['tokens'] + switch_tokens + minimux_tokens} (Switch: {switch_tokens} + 流程: {result['tokens']} + Mini Mux: {minimux_tokens})")


def test_trivia_batch() -> None:
    """批次測試所有 100 題，計算準確度"""
    if not OPENAI_API_KEY:
        raise RuntimeError("請先設定環境變數 OPENAI_API_KEY。")
    
    print("=== TRIVIA 批次測試 (100 題) ===\n")
    
    # 讀取題庫
    data = load_trivia_test_data()
    total_groups = len(data)
    
    model_fn = build_openai_model_fn(MODEL_NAME)
    mini_model_fn = build_openai_model_fn(MINI_MODEL_NAME)
    
    # 統計變數
    total_tokens = 0
    correct_count = 0
    total_questions = 0
    error_count = 0
    
    # 編譯正則表達式（省時間）
    puncts = re.compile(r'[^\w\s]')
    spaces = re.compile(r'\s+')
    stopwords = re.compile(r'\b(the|a|an|and|or|of|in|on|at|to|for|with|by)\b')
    
    def normalize_answer(ans: str) -> str:
        """正規化答案"""
        if not ans:
            return ""
        s = str(ans).lower().strip()
        s = puncts.sub('', s)
        s = stopwords.sub(' ', s)
        s = spaces.sub(' ', s).strip()
        return s
    
    print(f"開始測試 {total_groups} 組...\n")
    
    for idx, item in enumerate(data):
        topic = item.get("topic", "General")
        questions = item.get("questions", [])
        correct_answers_list = item.get("answers", [])
        
        # 計算即時準確度
        current_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # 進度條顯示
        percent = (idx / total_groups)
        filled = int(30 * percent)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\r進度: |{bar}| {idx}/{total_groups} ({percent*100:.1f}%) | 準確度: {current_accuracy:.1f}% | {topic[:20]:<20}", end="", flush=True)
        
        try:
            # 步驟 1: 生成人格群組
            switch_prompt = trivia_personas_switch_prompt.replace("{topic}", topic)
            switch_result, switch_tokens, _ = nano_run(persona="", user_text=switch_prompt, model_fn=model_fn)
            total_tokens += switch_tokens
            
            # 解析人格群組
            try:
                personas_data = json.loads(switch_result)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", switch_result, flags=re.S)
                personas_data = json.loads(m.group(0)) if m else {"groups": [{"group_id": 1, "thinker": "Essayist"}]}
            
            # 步驟 2: 執行多群組流程
            result = run_trivia_multi_group_flow(model_fn, topic, questions, personas_data, silent=True)
            total_tokens += result.get('tokens', 0)
            
    if "error" in result:
                error_count += 1
                continue
            
            # 取得 researcher 答案
            researcher_answers = result.get('answers', [])
            
            # 答案預處理：裁切到正確長度
            researcher_answers = researcher_answers[:len(questions)]
            
            # 步驟 3: Mini Mux（計入 token，但不影響準確度判斷）
            stories = []
            for group_id in sorted(result.get("final_results", {}).keys()):
                story = result["final_results"][group_id].get("final_story", "")
                stories.append(story if story else "")
            while len(stories) < 3:
                stories.append("")
            
            try:
                _, minimux_tokens, _ = mini_mux(researcher_answers, stories, trivia_minimux_prompt, mini_model_fn)
                total_tokens += minimux_tokens
            except:
                pass  # Mini Mux 失敗不影響準確度
            
            # 計算準確度（逐題比對）
            group_correct = 0
            for i, researcher_ans in enumerate(researcher_answers):
                researcher_norm = normalize_answer(researcher_ans)
                if not researcher_norm:
                    continue
                
                # 檢查是否匹配任何正確答案
                for correct_ans in correct_answers_list[i]:
                    correct_norm = normalize_answer(correct_ans)
                    # 精確匹配、包含匹配或關鍵詞匹配
                    if (researcher_norm == correct_norm or 
                        researcher_norm in correct_norm or 
                        correct_norm in researcher_norm):
                        group_correct += 1
                        break
                    # 關鍵詞匹配（長度>2的詞）
                    researcher_words = [w for w in researcher_norm.split() if len(w) > 2]
                    correct_words = [w for w in correct_norm.split() if len(w) > 2]
                    if any(w in correct_norm for w in researcher_words) or \
                       any(w in researcher_norm for w in correct_words):
                        group_correct += 1
                        break
            
            correct_count += group_correct
            total_questions += len(questions)
                
        except Exception as e:
            error_count += 1
    
    # 完成進度條
    bar = "█" * 30
    final_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    print(f"\r進度: |{bar}| {total_groups}/{total_groups} (100.0%) | 準確度: {final_accuracy:.1f}% | 完成!{' '*20}")
    
    # 最終統計
    print(f"\n{'='*50}")
    print(f"測試完成！")
    print(f"總題組: {total_groups} | 成功: {total_groups - error_count} | 錯誤: {error_count}")
    print(f"總問題數: {total_questions}")
    print(f"正確答案: {correct_count}")
    print(f"準確度: {final_accuracy:.1f}%")
    print(f"總 Token: {total_tokens:,} | 平均/組: {total_tokens/total_groups:.0f}")
    print(f"{'='*50}")


def main():
    """主程式入口"""
    print("請選擇測試模式:")
    print("1. trivia - 單題完整測試")
    print("2. batch - 批次測試（100題）\n")
    
    mode = input("請輸入 'trivia' 或 'batch': ").strip().lower()
    
    if mode == "trivia":
        test_trivia_full()
    elif mode == "batch":
        test_trivia_batch()
    else:
        print("無效選項，執行單題測試...")
        test_trivia_full()


if __name__ == "__main__":
    main()
