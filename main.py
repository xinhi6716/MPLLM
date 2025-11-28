import argparse
import time
import json
import os
import sys
import random

import utils.nesting_manager

# === [Imports] Pipelines ===
from pipeline_core import run_mpllm_pipeline 
import pipeline_nesting

# === [Imports] Utils & Core ===
from utils.api_client import get_openai_model_fn
from utils.data_loader import load_dataset
from utils.logger import save_batch_results
from utils.evaluator import evaluate_response
from core.tracker import CostTracker

def main():
    # === 0. 設定預設路徑 ===
    DATA_TRIVIA_PATH = "data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl"
    DATA_CODENAMES_PATH = "data/codenames_collaborative/codenames_50.jsonl"
    DATA_LOGIC_PATH = "data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl"

    # 1. 解析命令行參數
    parser = argparse.ArgumentParser(description="MPLLM Nano Runner")
    parser.add_argument('--task', type=str, choices=['trivia', 'codenames', 'logic'], help="Task to run")
    parser.add_argument('--mode', type=str, choices=['base', 'nesting'], default='base', help="Select pipeline architecture")
    parser.add_argument('--data', type=str, help="Path to .jsonl dataset")
    parser.add_argument('--limit', type=int, default=1, help="Number of items to test")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive chat mode")
    args = parser.parse_args()

    # ==========================================
    # 1.1 互動式選單 (若無參數輸入)
    # ==========================================
    if not args.task and not args.interactive:
        print("\n" + "="*45)
        print(" 🤖 MPLLM Launcher Menu")
        print("="*45)
        
        # --- 選擇任務 ---
        print("[Step 1] 請選擇任務:")
        print(" 1. 📝 Trivia (Creative Writing)")
        print(" 2. 🕵️  Codenames")
        print(" 3. 🧩 Logic Puzzle")
        print("-" * 45)
        print(" 4. 💬 自由對話模式 (Chat Mode)")
        print("="*45)
        
        choice = input("👉 請輸入選項 (1-4): ").strip()
        
        if choice == '1': args.task = 'trivia'
        elif choice == '2': args.task = 'codenames'
        elif choice == '3': args.task = 'logic'
        elif choice == '4': args.interactive = True
        else:
            print("⚠️ 無效選項，預設執行 Trivia")
            args.task = 'trivia'

        # --- 選擇 Pipeline 模式 ---
        if not args.interactive:
            print("\n" + "-"*45)
            print("[Step 2] 選擇 Pipeline 架構:")
            print(" 1. 🛠️  Baseline (Standard Pipeline)")
            print(" 2. 🧪 Nesting  (Experimental Pipeline)")
            m_choice = input("👉 請輸入選項 (1-2): ").strip()
            if m_choice == '2': 
                args.mode = 'nesting'
            else:
                args.mode = 'base'

        # --- 選擇 數量 ---
        if not args.interactive:
            print("\n" + "-"*45)
            print(f"[Step 3] 選擇 '{args.task}' 的執行模式:")
            print(" 1. 🎲 Random Single (隨機抽 1 題)")
            print(" 2. 📚 Sequential Batch (依序測 N 題)")
            
            mode_choice = input("👉 請輸入選項 (1-2): ").strip()
            
            if mode_choice == '2':
                try:
                    limit_input = input("   请输入要執行的題數 (例如 5, 10): ").strip()
                    args.limit = int(limit_input) if limit_input else 5
                except ValueError:
                    args.limit = 5
                print(f"   📚 模式: 依序執行前 {args.limit} 題")
            else:
                args.limit = 1
                print("   🎲 模式: 隨機抽取 1 題")

            time.sleep(0.5)

    # === 1.2 自動填入資料路徑 ===
    if args.task and not args.data:
        if args.task == 'trivia': args.data = DATA_TRIVIA_PATH
        elif args.task == 'codenames': args.data = DATA_CODENAMES_PATH
        elif args.task == 'logic': args.data = DATA_LOGIC_PATH
        print(f"📂 Auto-selected data: {args.data}")

    # 2. 初始化模型與參數
    ARCHITECTURE = "MPLLM"
    # [User Request] 使用動態模型名稱
    MODEL_NAME = f"GPT5-{args.mode.upper()}" 
    # [User Request] 測試模式判斷
    TEST_MODE = "Batch" if args.limit > 1 else "Single"

    mini_model = get_openai_model_fn("gpt-5-mini")
    nano_model = get_openai_model_fn("gpt-5-nano")
    models = {'mini': mini_model, 'nano': nano_model}
    tracker = CostTracker()
    batch_results = []

    # [User Request] 決定 Pipeline 函式
    if args.mode == 'nesting':
        print("\n🧪 Switching to NESTING Pipeline...")
        pipeline_run_fn = pipeline_nesting.run_pipeline
    else:
        print("\n🛠️ Using BASELINE Pipeline...")
        pipeline_run_fn = run_mpllm_pipeline

    # === 模式 A: 自由對話模式 ===
    if args.interactive:
        print("\n=== 💬 Interactive Chat Mode (No Scoring) ===")
        while True:
            try:
                user_q = input("\nUser Topic: ")
                if user_q.lower() in ['exit', 'quit']: break
                ans, _ = pipeline_run_fn('trivia', {"topic": user_q, "questions": []}, models, tracker)
                print(f"\n🤖 MPLLM: {ans}\n")
            except KeyboardInterrupt: break
            except Exception: break
        return

    # === 模式 B: 數據集評測模式 ===
    print(f"=== 🚀 Running: {args.task} | Mode: {TEST_MODE} | Limit: {args.limit} ===")
    
    dataset = load_dataset(args.task, args.data)
    if not dataset:
        print(f"❌ Error: Cannot load data from {args.data}")
        return

    # ==========================================
    # [核心] 題目選擇邏輯
    # ==========================================
    items_to_process = []
    if args.limit == 1:
        if len(dataset) > 0:
            selected = random.choice(dataset)
            original_idx = dataset.index(selected) + 1
            print(f"🎲 Randomly selected Item #{original_idx}")
            items_to_process = [selected]
    else:
        items_to_process = dataset[:args.limit]
        print(f"📚 Selected top {len(items_to_process)} items sequentially.")

    # ==========================================
    # 執行迴圈 (含進度條與即時存檔)
    # ==========================================
    total_score = 0.0
    total_items = len(items_to_process)
    bar_length = 30
    
    # 用於存檔的 Meta Info
    meta_info = {
        "architecture": ARCHITECTURE,
        "model": MODEL_NAME,
        "mode": TEST_MODE
    }

    print("\nProcessing...")

    for i, item in enumerate(items_to_process):
        start_time = time.time()
        
        try:
            # 執行 Pipeline
            final_ans, trace = pipeline_run_fn(args.task, item, models, tracker)
            duration = time.time() - start_time
            
            # 評分
            eval_result = evaluate_response(args.task, final_ans, item)
            score = eval_result.get('score', 0.0)
            details = eval_result.get('details', "")
            total_score += score
            
            # 獲取 Token
            run_tokens = trace.get('total_tokens', 0)
            
            # 格式化輸出
            ans_str = json.dumps(final_ans, ensure_ascii=False)
            
            # 記錄
            current_stats = tracker.get_summary()
            result_entry = {
                "id": i + 1,
                "task": args.task,
                "mode": args.mode,
                "input_summary": str(item)[:100].replace("\n", " "),
                "final_answer": ans_str,
                "tokens": run_tokens,
                "cost": current_stats['cost_usd'],
                "time": duration,
                "score": score,
                "eval_details": details
            }
            batch_results.append(result_entry)
            
            # [即時存檔]
            save_batch_results(batch_results, meta_info)

            # [進度條 UI]
            progress = (i + 1) / total_items
            filled = int(bar_length * progress)
            bar = '█' * filled + '-' * (bar_length - filled)
            avg_score = total_score / (i + 1)
            
            # 使用 \r 覆蓋當前行
            sys.stdout.write(f'\r|{bar}| {i+1}/{total_items} [{(progress*100):.0f}%] - Avg: {avg_score:.2f} - Last: {duration:.1f}s')
            sys.stdout.flush()
            
        except Exception as e:
            # 出錯時換行顯示，避免破壞進度條
            print(f"\n⚠️ Error on item {i+1}: {e}")
            # import traceback; traceback.print_exc()

    print("\n\n" + "="*45)
    final_avg = total_score / total_items if total_items > 0 else 0
    print(f"✅ Finished {total_items} items.")
    print(f"🏆 Final Avg Score: {final_avg:.2%}")
    print(f"💰 Total Cost: ${tracker.get_summary()['cost_usd']:.6f}")
    print(f"📂 Results saved to: logs/ (Model: {MODEL_NAME})")
    print("="*45)

if __name__ == "__main__":
    main()