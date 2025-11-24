# main.py
import argparse
import time
from utils.api_client import get_openai_model_fn
from utils.data_loader import load_dataset
from utils.logger import save_batch_results
from core.tracker import CostTracker
from pipeline_core import run_mpllm_pipeline

def main():
    # 1. 解析命令行參數
    parser = argparse.ArgumentParser(description="MPLLM Nano Runner")
    parser.add_argument('--task', type=str, choices=['trivia', 'codenames', 'logic'], help="Task to run")
    parser.add_argument('--data', type=str, help="Path to .jsonl dataset")
    parser.add_argument('--limit', type=int, default=1, help="Number of items to test")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive mode")
    args = parser.parse_args()

    # 2. 設定架構與模型參數 (用於檔名生成)
    ARCHITECTURE = "MPLLM"
    # 這裡你可以根據實際使用的模型修改，例如 "GPT4o-Mix"
    MODEL_NAME = "GPT4o-Mix" 
    TEST_MODE = "Batch" if args.limit > 1 else "Single"

    # 3. 初始化模型 (Dependency Injection)
    # 這裡混合使用了 nano 和 mini，目前都指向 gpt-4o-mini
    mini_model = get_openai_model_fn("gpt-4o-mini")
    nano_model = get_openai_model_fn("gpt-4o-mini")
    
    models = {'mini': mini_model, 'nano': nano_model}
    tracker = CostTracker()

    # 收集所有結果的容器
    batch_results = []

    # 4. 互動模式
    if args.interactive or (not args.task and not args.data):
        print("=== Interactive Mode ===")
        user_q = input("Question: ")
        item = {"topic": user_q, "questions": []}
        
        start_time = time.time()
        ans, trace = run_mpllm_pipeline('trivia', item, models, tracker)
        duration = time.time() - start_time
        
        print(f"Answer: {ans}")
        # 互動模式通常不寫入正式報表，或可視為 Single 測試
        return

    # 5. 數據集模式
    print(f"=== {ARCHITECTURE} Runner: {args.task} | Mode: {TEST_MODE} ===")
    dataset = load_dataset(args.task, args.data)
    if not dataset:
        print("❌ No data found.")
        return

    from utils.evaluator import evaluate_response  
    # 6. 批次執行
    total_score = 0.0
    
    for i, item in enumerate(dataset[:args.limit]):
        print(f"\n🚀 Processing Item {i+1}/{args.limit}...")
        
        # 開始計時
        start_time = time.time()
        
        # 執行 Pipeline
        final_ans, trace = run_mpllm_pipeline(args.task, item, models, tracker)
        
        # 結束計時
        duration = time.time() - start_time
        
        # === 新增：執行評分 ===
        eval_result = evaluate_response(args.task, final_ans, item)
        score = eval_result.get('score', 0)
        total_score += score
        
        print(f"🤖 Answer: {str(final_ans)[:60]}...") 
        print(f"⏱️  Time: {duration:.2f}s | 🏆 Score: {score:.2f} ({eval_result.get('details')})")
        
        current_stats = tracker.get_summary()
        
        result_entry = {
            "id": i + 1,
            "task": args.task,
            "input_summary": str(item)[:100].replace("\n", " "),
            "final_answer": str(final_ans)[:200].replace("\n", " "),
            "tokens": current_stats['total_tokens'],
            "cost": current_stats['cost_usd'],
            "time": duration,
            # 新增欄位
            "score": score,
            "eval_details": eval_result.get('details')
        }
        batch_results.append(result_entry)

    # 7. 輸出總表
    task_info = {
        "architecture": ARCHITECTURE,
        "model": MODEL_NAME,
        "mode": TEST_MODE
    }
    save_batch_results(batch_results, task_info)

    # 8. 終端機總結
    print("\n" + "="*50)
    avg_score = total_score / len(batch_results) if batch_results else 0
    print(f"✅ Completed {len(batch_results)} tasks.")
    print(f"🏆 Average Score: {avg_score:.2%}") # 顯示平均準確率
    print(f"💰 Total Accumulative Cost: ${tracker.get_summary()['cost_usd']:.6f}")

if __name__ == "__main__":
    main()