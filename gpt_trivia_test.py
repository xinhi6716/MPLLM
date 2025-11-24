import os
import json
import random
import re
import csv
import datetime
import time
from typing import Dict, Any, List, Tuple

from mpllm_prompts.prompts import gpt_trivia_prompt

# API 設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-nano")  # 對照組使用 Mini

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
                max_completion_tokens=5000,
            )
            text = (resp.choices[0].message.content or "").strip()
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


def create_trivia_prompt(questions: List[str], topic: str) -> str:
    """創建合併的 trivia prompt（回答問題 + 寫故事）"""
    questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = gpt_trivia_prompt.replace("{topic}", topic).replace("{questions}", questions_str)
    return prompt


def run_trivia_phase(model_fn, questions: List[str], topic: str, verbose: bool = False) -> Tuple[List[str], str, int]:
    """合併階段: 直接回答問題並撰寫故事
    
    Returns:
        (answers_list, story, tokens)
    """
    prompt = create_trivia_prompt(questions, topic)
    messages = [{"role": "user", "content": prompt}]
    
    result, tokens = model_fn(messages)
    
    # 解析結果
    answers_list = []
    story = ""
    
    try:
        # 嘗試直接解析 JSON
        data = json.loads(result)
        answers_list = data.get("answers", [])
        story = data.get("story", "").strip()
    except json.JSONDecodeError:
        # 嘗試用正則表達式提取 JSON
        m = re.search(r'\{[\s\S]*?"answers"[\s\S]*?"story"[\s\S]*?\}', result)
        if m:
            try:
                data = json.loads(m.group(0))
                answers_list = data.get("answers", [])
                story = data.get("story", "").strip()
            except json.JSONDecodeError:
                # 分別提取 answers 和 story
                # 提取 answers
                answers_match = re.search(r'"answers"[\s\S]*?:[\s\S]*?\[([^\]]+)\]', result)
                if answers_match:
                    answers_list = [a.strip().strip('"\'') for a in answers_match.group(1).split(',')]
                
                # 提取 story
                story_match = re.search(r'"story"[\s\S]*?:[\s\S]*?"([^"]+)"', result)
                if story_match:
                    story = story_match.group(1).strip()
    
    # 確保答案數量正確
    while len(answers_list) < len(questions):
        answers_list.append("unknown")
    answers_list = answers_list[:len(questions)]
    
    # 清理故事（移除可能的雙引號）
    if story.startswith('"') and story.endswith('"'):
        story = story[1:-1]
    if story.startswith("'") and story.endswith("'"):
        story = story[1:-1]
    
    if verbose:
        print(f"  答案: {answers_list}")
        print(f"  故事: {story[:100]}...")
        print(f"  Token 消耗: {tokens}")
    
    return answers_list, story, tokens


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


def test_single():
    """單題測試"""
    print("=== GPT-5 Mini 對照組測試 (單題) ===")
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

    # 記錄開始時間
    start_time = time.time()

    # 合併階段: 直接回答問題並撰寫故事
    print("執行: 回答問題並撰寫故事", end="", flush=True)
    answers_list, story, total_tokens = run_trivia_phase(model_fn, questions, topic, verbose=True)
    print(" ✓\n")

    # 記錄結束時間
    elapsed_time = time.time() - start_time

    # 總結
    print("=" * 50)
    print(f"\n📊 Token 統計:")
    print(f"   總計: {total_tokens:,} tokens")
    print(f"\n⏱️  執行時間: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分鐘)")

    # 答案比對
    correct_count = 0
    total_questions_count = len(questions)

    print(f"\n✅ 答案比對:")
    for i, final_ans in enumerate(answers_list):
        if i >= len(correct_answers_list):
            break

        final_norm = normalize_answer(final_ans)
        if not final_norm:
            status = "❌"
            print(f"   問題 {i+1}: {status} | 預測: {final_ans[:30]:<30} | 正確: {', '.join(correct_answers_list[i][:2])}")
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

    # 保存結果到 CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "results_gpt_trivia.csv")

    # 確保 CSV 有標頭
    header = ["時間", "主題", "問題數", "正確數", "準確率", "總 Token", "執行時間(秒)", "測試模式", "答案", "故事"]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(header)
    else:
        try:
            with open(csv_path, 'r+', encoding='utf-8-sig') as f:
                first_line = f.readline()
                if not first_line or "時間" not in first_line:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(",".join(header) + "\n" + content)
        except Exception:
            pass

    # 獲取當前時間戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 準備輸出內容
    answers_json = json.dumps(answers_list, ensure_ascii=False, separators=(',', ':'))

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, topic, len(questions), correct_count, f"{accuracy:.1f}%",
            total_tokens, f"{elapsed_time:.2f}", "single",
            answers_json, story
        ])

    print(f"\n📊 結果已儲存至: {csv_path}")
    print("=" * 50)


def test_batch():
    """批次測試所有 100 題"""
    print("=== GPT-5 Mini 對照組測試 (批次 100 題) ===\n")

    # 讀取題庫
    data = load_trivia_test_data()
    total_groups = len(data)

    model_fn = build_openai_model_fn(MODEL_NAME)

    # 統計變數
    total_tokens = 0
    correct_count = 0
    total_questions = 0
    error_count = 0

    # 時間統計
    total_elapsed_time = 0.0

    # 確保 results 資料夾存在
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # CSV 檔案路徑
    csv_path = os.path.join(results_dir, "results_gpt_trivia.csv")

    print(f"開始測試 {total_groups} 組...")
    print(f"結果將記錄至: {csv_path}\n")

    # 檢查並寫入 CSV header
    header = ["時間", "主題", "問題數", "正確數", "準確率", "總 Token", "執行時間(秒)", "測試模式", "答案", "故事"]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    else:
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline()
                if not first_line or "時間" not in first_line:
                    with open(csv_path, 'r+', encoding='utf-8-sig') as f2:
                        content = f2.read()
                        f2.seek(0, 0)
                        f2.write(",".join(header) + "\n" + content)
        except Exception:
            pass

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
        item_tokens = 0
        item_error = ""
        item_group_correct = 0
        item_error_occurred = False
        item_answers = []
        item_story = ""

        # 記錄單題開始時間
        item_start_time = time.time()

        try:
            # 合併階段: 直接回答問題並撰寫故事
            answers_list, story, item_tokens = run_trivia_phase(model_fn, questions, topic, verbose=False)
            total_tokens += item_tokens

            # 計算準確度
            group_correct = 0
            for i, final_ans in enumerate(answers_list):
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

            item_group_correct = group_correct
            correct_count += group_correct
            total_questions += len(questions)

            # 記錄輸出內容
            item_answers = answers_list
            item_story = story

        except Exception as e:
            error_count += 1
            item_error_occurred = True
            item_error = str(e)
            import traceback
            if error_count <= 3:  # 只顯示前3個錯誤
                traceback.print_exc()

        # 記錄單題結束時間
        item_elapsed_time = time.time() - item_start_time
        total_elapsed_time += item_elapsed_time

        # 立即寫入每一題的結果到 CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item_accuracy = (item_group_correct / len(questions) * 100) if questions else 0

        # 準備輸出內容
        answers_json = json.dumps(item_answers, ensure_ascii=False, separators=(',', ':'))

        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, topic, len(questions), item_group_correct, f"{item_accuracy:.1f}%",
                item_tokens, f"{item_elapsed_time:.2f}", "batch",
                answers_json, item_story
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

    # 成本估算（GPT-5 Mini 價格）
    input_ratio, output_ratio = 0.8, 0.2
    input_cost = total_tokens * input_ratio * 0.250 / 1_000_000
    output_cost = total_tokens * output_ratio * 2.000 / 1_000_000
    total_cost = input_cost + output_cost

    # 台幣換算 (1 USD = 30 TWD)
    total_cost_twd = total_cost * 30

    print(f"預估成本 (GPT-5 Mini):")
    print(f"   總計: ${total_cost:.6f} USD / NT${total_cost_twd:.2f}")
    print(f"平均成本/組: ${total_cost/total_groups:.6f} USD / NT${total_cost_twd/total_groups:.2f}")

    # 寫入批次測試總體統計到 CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, f"批次測試({total_groups}組)", total_questions, correct_count, f"{final_accuracy:.1f}%",
            total_tokens, f"{total_elapsed_time:.2f}", "batch_summary",
            "", ""  # 批次總統計不包含個別輸出內容
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
        test_batch()
    elif mode == "single":
        test_single()
    else:
        print("無效選項，執行單題測試...")
        test_single()


if __name__ == "__main__":
    main()

