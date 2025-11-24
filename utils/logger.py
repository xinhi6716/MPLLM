# utils/logger.py
import csv
import os
from datetime import datetime

def save_batch_results(results: list, task_info: dict):
    """
    將批次執行結果儲存為 CSV
    
    Args:
        results: 包含每筆測試結果的 list (dict)
        task_info: 包含架構資訊的 dict
            - architecture: "MPLLM" or "SPP"
            - model: "GPT4o", "GPT3.5" (主要模型前綴)
            - mode: "Batch" or "Single"
    """
    # 1. 準備目錄
    folder_name = "test_DATA"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 2. 生成檔名: 架構_模型_測試方式_日期.csv
    # 例如: MPLLM_GPT4o_Batch_2023-10-27.csv
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{task_info['architecture']}_{task_info['model']}_{task_info['mode']}_{date_str}.csv"
    filepath = os.path.join(folder_name, filename)
    
    # 3. 定義欄位 (比原本更豐富)
    fieldnames = [
        "Run_ID",
        "Task_Type",
        "Timestamp",
        "Input_Summary",
        "Final_Answer",
        "Score",            # <--- 新增
        "Eval_Details",     # <--- 新增
        "Total_Tokens",
        "Cost_USD",
        "Execution_Time_Sec",
        "Status"
    ]
    
    # 4. 寫入檔案 (使用 append 模式，這樣同一天的測試會累積在同一個檔)
    file_exists = os.path.isfile(filepath)
    
    try:
        with open(filepath, mode='a', newline='', encoding='utf-8-sig') as f: # utf-8-sig 讓 Excel 開啟不亂碼
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            for res in results:
                # 整理每筆資料
                row = {
                    "Run_ID": res.get("id"),
                    "Task_Type": res.get("task"),
                    "Score": res.get("score", 0),                 # <--- 寫入分數
                    "Eval_Details": res.get("eval_details", ""),  # <--- 寫入詳情
                    "Total_Tokens": res.get("tokens", 0),
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Input_Summary": res.get("input_summary"),
                    "Final_Answer": res.get("final_answer"),
                    "Total_Tokens": res.get("tokens", 0),
                    "Cost_USD": round(res.get("cost", 0), 6),
                    "Execution_Time_Sec": round(res.get("time", 0), 2),
                    "Status": "Success" if res.get("final_answer") else "Fail"
                }
                writer.writerow(row)
                
        print(f"\n📊 [Report] Summary saved to: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Error saving log: {e}")

# 為了相容舊程式碼，保留原本的 log_to_csv 但指向新邏輯 
def log_to_csv(data, filename="legacy_log.csv"):
    # 簡單的轉接，或是你可以直接廢棄它
    pass