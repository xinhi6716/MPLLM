import argparse
import time
import json
import os
import sys
import random  # <--- [新增] 引入 random
from utils.api_client import get_openai_model_fn
from utils.data_loader import load_dataset
from utils.logger import save_batch_results
from utils.evaluator import evaluate_response
from core.tracker import CostTracker
from pipeline_core import run_mpllm_pipeline

def main():
    # === 0. 設定預設路徑 ===
    DATA_TRIVIA_PATH = "data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl"
    DATA_CODENAMES_PATH = "data/codenames_collaborative/codenames_50.jsonl"
    DATA_LOGIC_PATH = "data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl"

    # 1. 解析命令行參數
    parser = argparse.ArgumentParser(description="MPLLM Nano Runner")
    parser.add_argument('--task', type=str, choices=['trivia', 'codenames', 'logic'], help="Task to run")
    parser.add_argument('--data', type=str, help="Path to .jsonl dataset")
    parser.add_argument('--limit', type=int, default=1, help="Number of items to test")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive chat mode")
    args = parser.parse_args()

    # ==========================================
    # 互動式選單 (當沒有指定 task 時觸發)
    # ==========================================
    if not args.task and not args.interactive:
        print("\n" + "="*45)
        print(" 🤖 MPLLM Launcher Menu")
        print("="*45)
        
        # --- 步驟 1: 選擇任務 ---
        print("[Step 1] 請選擇任務:")
        print(" 1. 📝 Trivia (Creative Writing)")
        print(" 2. 🕵️  Codenames")
        print(" 3. 🧩 Logic Puzzle")
        print("-" * 45)
        print(" 4. 💬 自由對話模式 (Chat Mode - No Scoring)")
        print("="*45)
        
        choice = input("👉 請輸入選項 (1-4): ").strip()
        
        if choice == '1': args.task = 'trivia'
        elif choice == '2': args.task = 'codenames'
        elif choice == '3': args.task = 'logic'
        elif choice == '4': args.interactive = True
        else:
            print("⚠️ 無效選項，預設執行 Trivia")
            args.task = 'trivia'

        # --- 步驟 2: 選擇 Single 或 Batch (若非對話模式) ---
        if not args.interactive:
            print("\n" + "-"*45)
            print(f"[Step 2] 選擇 '{args.task}' 的執行模式:")
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

    # === 自動填入資料路徑 ===
    if args.task and not args.data:
        if args.task == 'trivia':
            args.data = DATA_TRIVIA_PATH
        elif args.task == 'codenames':
            args.data = DATA_CODENAMES_PATH
        elif args.task == 'logic':
            args.data = DATA_LOGIC_PATH
        print(f"📂 Auto-selected data: {args.data}")

    # 2. 初始化模型與參數
    ARCHITECTURE = "MPLLM"
    MODEL_NAME = "GPT5-Mix"
    TEST_MODE = "Batch" if args.limit > 1 else "Single"

    mini_model = get_openai_model_fn("gpt-5-mini")
    nano_model = get_openai_model_fn("gpt-5-nano")
    models = {'mini': mini_model, 'nano': nano_model}
    tracker = CostTracker()
    batch_results = []

    # === 模式 A: 自由對話模式 ===
    if args.interactive:
        print("\n=== 💬 Interactive Chat Mode (No Scoring) ===")
        while True:
            try:
                user_q = input("\nUser Topic: ")
                if user_q.lower() in ['exit', 'quit']: break
                ans, _ = run_mpllm_pipeline('trivia', {"topic": user_q, "questions": []}, models, tracker)
                print(f"\n🤖 MPLLM: {ans}\n")
            except KeyboardInterrupt: break
            except Exception: break
        return

    # === 模式 B: 數據集評測模式 ===
    print(f"=== 🚀 Running: {args.task} | Mode: {TEST_MODE} ===")
    
    dataset = load_dataset(args.task, args.data)
    if not dataset:
        print(f"❌ Error: Cannot load data from {args.data}")
        return

    # ==========================================
    # [核心修改] 題目選擇邏輯
    # ==========================================
    items_to_process = []
    
    if args.limit == 1:
        # Single Mode: 隨機選一題
        if len(dataset) > 0:
            selected = random.choice(dataset)
            # 嘗試找出它是原始資料集中的第幾題 (index+1)
            original_idx = dataset.index(selected) + 1
            print(f"🎲 Randomly selected Item #{original_idx} (from {len(dataset)} items)")
            items_to_process = [selected]
    else:
        # Batch Mode: 選前 N 題
        items_to_process = dataset[:args.limit]
        print(f"📚 Selected top {len(items_to_process)} items sequentially.")

    # ==========================================
    # 執行迴圈
    # ==========================================
    total_score = 0.0
    processed_count = 0
    
    for i, item in enumerate(items_to_process):
        # 顯示當前進度 (如果是隨機，這裡的 i+1 只是執行次序)
        print(f"\n🔸 Processing Task {i+1}/{len(items_to_process)}...")
        start_time = time.time()
        
        try:
            final_ans, trace = run_mpllm_pipeline(args.task, item, models, tracker)
            duration = time.time() - start_time
            
            eval_result = evaluate_response(args.task, final_ans, item)
            score = eval_result.get('score', 0.0)
            details = eval_result.get('details', "")
            
            total_score += score
            processed_count += 1
            
            # 顯示簡化結果
            ans_str = json.dumps(final_ans, ensure_ascii=False)
            display_str = (ans_str[:75] + '...') if len(ans_str) > 75 else ans_str
            
            print(f"   🤖 Output: {display_str}") 
            print(f"   🏆 Score: {score:.2f} ({details}) | ⏱️ {duration:.2f}s")
            
            current_stats = tracker.get_summary()
            result_entry = {
                "id": i + 1,
                "task": args.task,
                "input_summary": str(item)[:100].replace("\n", " "),
                "final_answer": ans_str,
                "tokens": trace.get('tokens', 0) if 'tokens' in trace else current_stats.get('total_tokens', 0),
                "cost": current_stats['cost_usd'],
                "time": duration,
                "score": score,
                "eval_details": details
            }
            batch_results.append(result_entry)
            
        except Exception as e:
            print(f"⚠️ Error on item: {e}")
            # import traceback; traceback.print_exc()

    if batch_results:
        save_batch_results(batch_results, {
            "architecture": ARCHITECTURE,
            "model": MODEL_NAME,
            "mode": TEST_MODE
        })

    print("\n" + "="*45)
    avg_score = total_score / processed_count if processed_count > 0 else 0
    print(f"✅ Finished {processed_count} items.")
    print(f"🏆 Avg Score: {avg_score:.2%}")
    print(f"💰 Total Cost: ${tracker.get_summary()['cost_usd']:.6f}")
    print("="*45)

if __name__ == "__main__":
    main()