# utils/data_loader.py
import json
import os

def load_dataset(task_type: str, file_path: str):
    """
    讀取特定任務的資料集 (.jsonl 或 .txt)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(_parse_item(task_type, item))
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON line in {file_path}")
        elif file_path.endswith('.txt'):
            for line in f:
                if line.strip():
                    data.append({"topic": line.strip()})
                    
    print(f"📦 Loaded {len(data)} items for task '{task_type}' from {file_path}")
    return data

def _parse_item(task_type, item):
    """標準化資料格式"""
    if task_type == 'trivia':
        return {
            "topic": item.get("topic", ""),
            "questions": item.get("questions", []),
            # 關鍵修正：加入 answers 欄位
            "answers": item.get("answers", []),
            "n": len(item.get("questions", []))
        }
    elif task_type == 'codenames':
        return {
            "target_words": item.get("target_words", []),
            "word_list": item.get("word_list", []),
            "n": len(item.get("target_words", []))
        }
    elif task_type == 'logic':
        return {
            "inputs": item.get("inputs", ""),
            "targets": item.get("targets", [])
        }
    return item