# utils/logger.py
import csv
import time
import os

def log_to_csv(data: dict, filename="mpllm_experiment_log.csv"):
    file_exists = os.path.isfile(filename)
    
    # 確保 timestamp 存在
    if 'timestamp' not in data:
        data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    print(f"✔️ Log saved to {filename}")