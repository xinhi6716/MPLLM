# main.py
import os
from utils.api_client import get_openai_model_fn
from utils.logger import log_to_csv
from core.tracker import CostTracker
from pipeline_core import run_mpllm_pipeline

def main():
    # 1. 設定 API Key (請確認環境變數或直接填入)
    # os.environ["OPENAI_API_KEY"] = "sk-..." 
    
    # 2. 準備依賴注入 (Dependency Injection)
    # 我們可以給不同層不同的模型設定
    try:
        mini_model = get_openai_model_fn(model_name="gpt-4o-mini")
        # 假設我們想用同一個模型模擬 nano
        nano_model = get_openai_model_fn(model_name="gpt-4o-mini")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    models = {
        'mini': mini_model,
        'nano': nano_model
    }
    
    tracker = CostTracker()
    
    # 3. 測試輸入
    user_query = input("請輸入您的問題 (或按 Enter 使用預設測試題): ")
    if not user_query:
        user_query = "解釋量子糾纏如何應用於未來的加密技術，並舉一個生活化的例子。"

    print(f"\n🚀 Starting MPLLM for: {user_query}\n" + "="*50)

    # 4. 執行流水線
    final_answer, trace_data = run_mpllm_pipeline(user_query, models, tracker)

    # 5. 顯示結果
    print("="*50)
    print("🤖 Final Answer:\n")
    print(final_answer)
    print("="*50)
    
    # 6. 結算與記錄
    stats = tracker.get_summary()
    print(f"💰 Cost: ${stats['cost_usd']} | Tokens: {stats['total_tokens']}")
    
    log_data = {
        "input": user_query,
        "final_answer": final_answer,
        "total_tokens": stats['total_tokens'],
        "cost_usd": stats['cost_usd']
    }
    log_to_csv(log_data)

if __name__ == "__main__":
    main()